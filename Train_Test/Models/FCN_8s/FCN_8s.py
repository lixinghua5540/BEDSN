# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from Parameters_Set import numclasses

def print_layer(t):
    print(t.op.name, ' ', t.get_shape().as_list(), '\n')


def conv(x, d_out, name,  xavier=False):
    d_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if not xavier:
            kernel = tf.Variable(tf.truncated_normal([3, 3, d_in, d_out], stddev=0.1), name='weights')
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                               trainable=True,
                               name='bias')
        else:
            kernel = tf.get_variable(scope + 'weights', shape=[3, 3, d_in, d_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[d_out]),
                               trainable=True,
                               name='bias')
        conv = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(conv + bias, name=scope)
        return activation


def maxpool(x, name):
    activation = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name=name)
    print_layer(activation)
    return activation


def fc(x, n_out, name, xavier=False):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if not xavier:
            weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                               trainable=True,
                               name='bias')
            #print("truncated_normal")
        else:
            weight = tf.get_variable(scope + 'weights', shape=[n_in, n_out],
                                     dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]),
                               trainable=True,
                               name='bias')
        activation = tf.nn.relu_layer(x, weight, bias, name=name)
        print_layer(activation)
        return activation


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def FCN_8s(images,inference=True):
    conv1_1 = conv(images, 64, 'conv1_1')
    conv1_2 = conv(conv1_1, 64, 'conv1_2')
    pool1 = maxpool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 128, 'conv2_1')
    conv2_2 = conv(conv2_1, 128, 'conv2_2')
    pool2 = maxpool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 256, 'conv3_1')
    conv3_2 = conv(conv3_1, 256, 'conv3_2')
    conv3_3 = conv(conv3_2, 256, 'conv3_3')
    pool3 = maxpool(conv3_3, 'pool3')

    conv4_1 = conv(pool3, 512, 'conv4_1')
    conv4_2 = conv(conv4_1, 512, 'conv4_2')
    conv4_3 = conv(conv4_2, 512, 'conv4_3')
    pool4 = maxpool(conv4_3, 'pool4')

    conv5_1 = conv(pool4, 512, 'conv5_1')
    conv5_2 = conv(conv5_1, 512, 'conv5_2')
    conv5_3 = conv(conv5_2, 512, 'conv5_3')
    pool5 = maxpool(conv5_3, 'pool5')

    W6 = weight_variable([8, 8, 512, 4096], name="W6")
    b6 = bias_variable([4096], name="b6")
    conv6 = conv2d_basic(pool5, W6, b6)
    relu6 = tf.nn.relu(conv6, name="relu6")

    W7 = weight_variable([1, 1, 4096, 4096], name="W7")
    b7 = bias_variable([4096], name="b7")
    conv7 = conv2d_basic(relu6, W7, b7)
    relu7 = tf.nn.relu(conv7, name="relu7")

    W8 = weight_variable([1, 1, 4096, numclasses], name="W8")
    b8 = bias_variable([numclasses], name="b8")
    conv8 = conv2d_basic(relu7, W8, b8)

    deconv_shape1 = pool4.get_shape()
    W_t1 = weight_variable([4, 4, deconv_shape1[3].value, numclasses], name="W_t1")
    b_t1 = bias_variable([deconv_shape1[3].value], name="b_t1")
    conv_t1 = conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(pool4))
    fuse_1 = tf.add(conv_t1, pool4, name="fuse_1")

    deconv_shape2 = pool3.get_shape()
    W_t2 = weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
    b_t2 = bias_variable([deconv_shape2[3].value], name="b_t2")
    conv_t2 = conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(pool3))
    fuse_2 = tf.add(conv_t2, pool3, name="fuse_2")

    shape = tf.shape(images)
    deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], numclasses])
    W_t3 = weight_variable([16, 16, numclasses, deconv_shape2[3].value], name="W_t3")
    b_t3 = bias_variable([numclasses], name="b_t3")
    conv_t3 = conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

    annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return conv_t3


# if __name__ == "__main__":
#     inputs = tf.random.normal([1, 256, 256, 3])
#     result = FCN_8s(inputs, tf.constant(False,dtype=tf.bool))
#     print(result.get_shape().as_list())