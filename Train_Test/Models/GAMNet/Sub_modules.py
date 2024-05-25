# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf

def GateModule(f_high,f_low,filter_n,is_training):
    """
    利用作用在高级特征上的信息熵控制低级特征的权重，以指导低级特征的传播：实现更好地恢复边界的信息特征
    """
    f_high=tf.layers.conv2d(inputs=f_high,
                         filters=filter_n,
                         kernel_size=(1,1),
                         kernel_initializer=tf.variance_scaling_initializer(),
                         padding='SAME',
                         activation=None)
    f_high=tf.layers.batch_normalization(inputs=f_high,
                                   training=is_training,
                                   center=True,
                                   scale=True,
                                   fused=True)
    f_high = tf.nn.relu(f_high)   # 又添加了一层conv-bn-relu
    f_high_softmax=tf.nn.softmax(f_high,axis=3)
    f_high_G=-tf.reduce_sum((f_high_softmax* tf.log(f_high_softmax)),axis=3)
    f_high_G=tf.expand_dims(f_high_G,axis=3)
    # 信息熵
    F=tf.multiply(f_high_G,f_low)+f_high
    return F
def AttentionModule(f_high,f_low,filter_n,is_training):
    # filter_channel 是不是为2？？？
    f_high = tf.layers.conv2d(inputs=f_high,
                              filters=filter_n,
                              kernel_size=(1, 1),
                              kernel_initializer=tf.variance_scaling_initializer(),
                              padding='SAME',
                              activation=None)
    f_high = tf.layers.batch_normalization(inputs=f_high,
                                           training=is_training,
                                           center=True,
                                           scale=True,
                                           fused=True)
    f_high = tf.nn.relu(f_high)  # 又添加了一层conv-bn-relu
    f_low_high=tf.concat([f_high,f_low],axis=3)  # 拼接
    h_1=tf.layers.conv2d(inputs=f_low_high,
                         filters=2,
                         kernel_size=(3, 3),
                         kernel_initializer=tf.variance_scaling_initializer(),
                         padding='SAME',
                         activation=None)
    h_1 = tf.layers.batch_normalization(inputs=h_1,
                                      training=is_training,
                                      center=True,
                                      scale=True,
                                      fused=True)
    h_1=tf.nn.relu(h_1)

    h_2=tf.layers.conv2d(inputs=h_1,
                           filters=2, # 考虑到计算权重，滤波器为2，第一个也是2吗？？？
                           kernel_size=(3, 3),
                           kernel_initializer=tf.variance_scaling_initializer(),
                           padding='SAME',
                           activation=None)
    h_2=tf.layers.batch_normalization(inputs=h_2,
                                    training=is_training,
                                    center=True,
                                    scale=True,
                                    fused=True)
    h_2=tf.nn.relu(h_2)    # 两层注意力模型的输出H (两个卷积层)

    numerator_=tf.math.exp(h_2)  # W的分子
    denominator_=tf.reduce_sum(numerator_,axis=3)
    denominator_=tf.expand_dims(denominator_,axis=3)

    W=tf.div(numerator_,denominator_)   # 权重的计算
    W_0=tf.expand_dims(W[:, :, :, 0], axis=3)
    W_1=tf.expand_dims(W[:, :, :, 1], axis=3)
    A=tf.multiply(W_0, f_high)+tf.multiply(W_1, f_low)  # 作用同下式子所示
    # A=tf.multiply(W_0, tf.expand_dims(f_low_high[0],axis=0)) + tf.multiply(W_1,  tf.expand_dims(f_low_high[1],axis=0))
       #此处修改错误，因为f_high与f_low 进行拼接时是沿着第四维度axis=3的(channel1=128 channel2=128-->channel=256)。。。


    return A
def GAM(f_high,f_low,filter_n,is_training):   # f_high：解码端  f_low:编码端
    F_input=GateModule(f_high,f_low,filter_n,is_training)
    A_input=AttentionModule(f_high,f_low,filter_n,is_training)
    output_1=tf.concat([F_input,A_input],axis=3)
    output_1=tf.layers.conv2d(inputs=output_1,
                              filters=filter_n,
                              kernel_size=(3,3),
                              padding="SAME",
                              activation=None)
    output_1=tf.layers.batch_normalization(inputs=output_1,
                                           training=is_training,
                                           scale=True,
                                           fused=True)
    output_1=tf.nn.relu(output_1)
    return output_1


def Atrous_spatial_pyramid_pooling(inputs, filters=128,A_training=True):  # ASPP层
    '''
    Atrous Spatial Pyramid Pooling (ASPP) Block
    '''
    pool_height = tf.shape(inputs)[1]
    pool_width = tf.shape(inputs)[2]

    resize_height = pool_height
    resize_width = pool_width

    # Atrous Spatial Pyramid Pooling
    # Atrous 1x1
    aspp1x1 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1, 1), padding='same',
                               kernel_initializer=tf.variance_scaling_initializer(),name='aspp1x1')
    aspp1x1=tf.layers.batch_normalization(inputs= aspp1x1,training=A_training,center=True,scale=True, fused=True)

    aspp3x3_1=tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),padding='same',
                               dilation_rate=2, kernel_initializer=tf.variance_scaling_initializer(), name='aspp3x3_1')

    aspp3x3_1 = tf.layers.batch_normalization(inputs= aspp3x3_1, training=A_training, center=True, scale=True, fused=True)

    aspp3x3_2 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), padding='same',
                                 dilation_rate=5, kernel_initializer=tf.variance_scaling_initializer(), name='aspp3x3_2')

    aspp3x3_2 = tf.layers.batch_normalization(inputs= aspp3x3_2 , training=A_training, center=True, scale=True, fused=True)

    aspp3x3_3 = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3),padding='same',
                                 dilation_rate=8, kernel_initializer=tf.variance_scaling_initializer(),name='aspp3x3_3')
    aspp3x3_3 = tf.layers.batch_normalization(inputs= aspp3x3_3 , training=A_training, center=True,scale=True,fused=True)


    # Image Level Pooling
    image_feature = tf.reduce_mean(inputs, [1, 2], keepdims=True)
    image_feature = tf.layers.conv2d(inputs=image_feature, filters=filters, kernel_size=(1, 1),
                                     padding='same',kernel_initializer=tf.variance_scaling_initializer())
    image_feature  = tf.layers.batch_normalization(inputs=image_feature , training=A_training, center=True,scale=True, fused=True)

    image_feature = tf.image.resize_bilinear(images=image_feature,
                                             size=[resize_height, resize_width],
                                             align_corners=True, name='image_pool_feature')

    # Merge Poolings
    outputs = tf.concat(values=[aspp1x1, aspp3x3_1, aspp3x3_2, aspp3x3_3, image_feature],
                        axis=3, name='aspp_pools')

    outputs =tf.layers.conv2d(inputs=outputs, filters=filters, kernel_size=(1, 1),
                               padding='same', kernel_initializer=tf.variance_scaling_initializer(), name='aspp_outputs')
    return outputs
