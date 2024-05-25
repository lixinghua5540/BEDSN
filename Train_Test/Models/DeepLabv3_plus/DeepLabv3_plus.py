# -*- coding: utf-8 -*-
"""
@Time : 2022/6/19 23:09
@Author : wcf
@Email : lwang162022@qq.com
@File : DeepLabv3_plus.py
@Software: PyCharm
"""
import  os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
from Semantic_Segmentation.Common_Codes.Train_Test.Parameters_Set import numclasses

from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers


_BATCH_NORM_DECAY = 0.9997
_WEIGHT_DECAY = 5e-4

def atrous_spatial_pyramid_pooling(inputs, output_stride,is_training, batch_norm_decay=0.9997,depth=256):
  """Atrous Spatial Pyramid Pooling.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
      the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    is_training: A boolean denoting whether the input is for training.
    depth: The depth of the ResNet unit output.

  Returns:
    The atrous spatial pyramid pooling output.
  """
  with tf.variable_scope("aspp"):
    if output_stride not in [8, 16]:
      raise ValueError('output_stride must be either 8 or 16.')
    atrous_rates = [6,12,18]
    if output_stride == 8:
      atrous_rates=[2*rate for rate in atrous_rates]

    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      with arg_scope([layers.batch_norm], is_training=is_training):
        inputs_size = tf.shape(inputs)[1:3]
        # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        # the rates are doubled when output stride = 8.
        conv_1x1 = tf.layers.conv2d(inputs,depth,[1,1],name="conv_1x1")
        conv_3x3_1=tf.layers.conv2d(inputs,depth,[3,3],strides=1, dilation_rate=atrous_rates[0], name='conv_3x3_1',padding='same')
        conv_3x3_2=tf.layers.conv2d(inputs, depth, [3, 3], strides=1, dilation_rate=atrous_rates[1], name='conv_3x3_2',padding='same')
        conv_3x3_3=tf.layers.conv2d(inputs, depth, [3, 3], strides=1, dilation_rate=atrous_rates[2], name='conv_3x3_3',padding='same')

        # (b) the image-level features
        with tf.variable_scope("image_level_features"):
          # global average pooling
          image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling', keepdims=True)
          # 1x1 convolution with 256 filters( and batch normalization)
          image_level_features = tf.layers.conv2d(image_level_features, depth, [1, 1], strides=1, name='conv_1x1')
          # bilinearly upsample features
          image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size, name='upsample')

        net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
        net = tf.layers.conv2d(net, depth, [1, 1], strides=1, name='conv_1x1_concat')

        return net

def DeepLabv3_plus(inputs,
                  is_training,
                  num_classes=6,
                  output_stride=16,
                  base_architecture='resnet_v2_101',
                  batch_norm_decay=0.9997):
    if batch_norm_decay is None:
        batch_norm_decay = _BATCH_NORM_DECAY
    if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
        raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'.")
    if base_architecture == 'resnet_v2_50':
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101
    with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
      logits,end_points =base_model(inputs,
                                  num_classes=num_classes,
                                  is_training=is_training,
                                  global_pool=False,
                                  output_stride=output_stride)


    inputs_size = tf.shape(inputs)[1:3]
    net = end_points[base_architecture + '/block4']
    encoder_output=atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

    with tf.variable_scope("decoder"):
      with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        with arg_scope([layers.batch_norm], is_training=is_training):
          with tf.variable_scope("low_level_features"):
            low_level_features=end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
            low_level_features=layers_lib.conv2d(low_level_features, 48,[1, 1], stride=1, scope='conv_1x1')
            low_level_features_size = tf.shape(low_level_features)[1:3]

          with tf.variable_scope("upsampling_logits"):
            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
            net = tf.concat([net, low_level_features], axis=3, name='concat')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='conv_1x1')
            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')
    return   logits


if __name__ == "__main__":
    inputs = tf.random.normal([1, 256, 256, 3])
    result = DeepLabv3_plus(inputs, tf.constant(False,dtype=tf.bool))
    print(result.get_shape().as_list())