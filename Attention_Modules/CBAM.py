# -*- coding: utf-8 -*-

import tensorflow as tf
def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)
    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    return scale, input_feature * scale
def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()

    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2
        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')
    return concat, input_feature * concat
def channel_spatial_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    通道-空间注意力模块CBAM
    """
    with tf.variable_scope(name):
        scale, attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        concat, attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature
