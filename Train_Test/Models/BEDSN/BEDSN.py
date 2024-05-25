# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from Parameters_Set import numclasses

def CONV_BN_ReLU(input,filter,kesize,stride,is_training,name_layer):
    input=tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kesize,strides=stride,padding="SAME",use_bias=False,
                           kernel_initializer=tf.variance_scaling_initializer(),name=name_layer+"_conv")
    input=tf.layers.batch_normalization(inputs=input,training=is_training,name=name_layer+'_BN')
    input=tf.nn.relu(input,name=name_layer+'_relu')
    return input
def Inception_block(input,is_training,filters,name_layer):
    filters_name=list(filters.keys())
    assert filters_name==["1x1",'3x3','5x5','7x7'],"NameError in Inception module."
    Branch_1x1=CONV_BN_ReLU(input,filter=filters['1x1'],kesize=1,stride=1,is_training=is_training,name_layer=name_layer+'Inception_1x1')

    Branch_3x3=CONV_BN_ReLU(input,filter=filters['3x3'][0],kesize=1,stride=1,is_training=is_training,name_layer=name_layer+"Inception_3x3_conv1")
    Branch_3x3=CONV_BN_ReLU(input=Branch_3x3,filter=filters['3x3'][1],kesize=3,stride=1,is_training=is_training,name_layer=name_layer+"Inception_3x3")

    Branch_5x5=CONV_BN_ReLU(input,filter=filters['5x5'][0],kesize=1,stride=1,is_training=is_training,name_layer=name_layer+"Inception_5x5_conv1")
    Branch_5x5=CONV_BN_ReLU(input=Branch_5x5,filter=filters['5x5'][1],kesize=5,stride=1,is_training=is_training,name_layer=name_layer+"Inception_5x5")

    Branch_7x7=CONV_BN_ReLU(input,filter=filters['7x7'][0],kesize=1,stride=1,is_training=is_training,name_layer=name_layer+"Inception_7x7_conv1")
    Branch_7x7=CONV_BN_ReLU(input=Branch_7x7,filter=filters['7x7'][1],kesize=7,stride=1,is_training=is_training,name_layer=name_layer+"Inception_7x7")

    result=tf.concat([Branch_1x1,Branch_3x3,Branch_5x5,Branch_7x7],axis=-1)
    assert result.get_shape().as_list()[-1]==2*filters["3x3"][1],"ShapeError in Inception module."
    return result

def channel_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """
    with tf.variable_scope(name):
        scale, attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        output=tf.add(input_feature,attention_feature)
    return output

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

def BN(inputs,training):
    return tf.layers.batch_normalization(inputs=inputs,training=training,center=True,scale=True,fused=True)

def conv_3x3(inputs,filters):
    return tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=(3, 3),
                            kernel_initializer=tf.variance_scaling_initializer(),
                            padding='SAME',activation=None)
def conv_3x3_atrous(inputs,filters,atrous_rate):
    return tf.layers.conv2d(inputs=inputs,filters=filters,kernel_size=(3, 3),
                            dilation_rate=atrous_rate,
                            kernel_initializer=tf.variance_scaling_initializer(),
                            padding='SAME',activation=None)

def conv_1x1(inputs,filters):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1,1),
                            kernel_initializer=tf.variance_scaling_initializer(),
                            padding='SAME', activation=None)

def relu_(input):
    return tf.nn.relu(input)

def maxpool_2x2(inputs):
    return tf.layers.max_pooling2d(inputs=inputs,pool_size=(2, 2),strides=(2, 2))

def conv_transpose(inputs,strides):
    return tf.layers.conv2d_transpose(inputs=inputs,filters=1,kernel_size=strides,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      strides=strides, padding='valid',activation=None)
def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
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

def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1]*=2
        output_shape[2]*=2
        output_shape[3]=W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def BEDSN(input, is_training=True):#is training 不能设置为false
    _CHANNELS=[64,128,256,512,512]
    convs=[]
    """第一层"""
    with tf.name_scope("Encoder-layer1"):
        a10=relu_(BN(conv_3x3(input,_CHANNELS[0]),training=is_training))
        a11=relu_(BN(conv_3x3(a10,_CHANNELS[0]),training=is_training))
    convs.append(a11)

    """第二层"""
    with tf.name_scope("Encoder-layer2"):
        a20=maxpool_2x2(a11)
        a20=relu_(BN(conv_3x3(a20,_CHANNELS[1]),training=is_training))
        a21=relu_(BN(conv_3x3(a20,_CHANNELS[1]),training=is_training))
    convs.append(a21)

    """第三层"""
    with tf.name_scope("Encoder-layer3"):
        a30=maxpool_2x2(a21)
        a30=relu_(BN(conv_3x3(a30, _CHANNELS[2]), training=is_training))
        a31=relu_(BN(conv_3x3(a30, _CHANNELS[2]), training=is_training))
        a32=relu_(BN(conv_3x3(a31, _CHANNELS[2]), training=is_training))
    convs.append(a32)

    """第四层"""
    with tf.name_scope("Encoder-layer4"):
        a40=maxpool_2x2(a32)
        a40= Inception_block(input=a40, is_training=is_training,filters={"1x1": 64, "3x3": [192, 256], "5x5": [64, 128], "7x7": [32, 64]}, name_layer="Layer4_stage1")
        a41= Inception_block(input=a40, is_training=is_training,filters={"1x1": 64, "3x3": [192, 256], "5x5": [64, 128], "7x7": [32, 64]},name_layer="Layer4_stage2")
        a42= Inception_block(input=a41, is_training=is_training,filters={"1x1": 64, "3x3": [192, 256], "5x5": [64, 128], "7x7": [32, 64]},name_layer="Layer4_stage3")
        assert a42.get_shape().as_list()[-1] == _CHANNELS[-1], "ChannelMatch Error in Inception_D."
    convs.append(a42)

    """第五层"""   # ---------------------------------
    with tf.name_scope("Encoder-layer5"):
        a50=maxpool_2x2(a42)
        a50=relu_(BN(conv_3x3_atrous(a50, _CHANNELS[4],1), training=is_training))
        a51=relu_(BN(conv_3x3_atrous(a50, _CHANNELS[4],2), training=is_training))
        a52=relu_(BN(conv_3x3_atrous(a51, _CHANNELS[4],3), training=is_training))
    convs.append(a52)

    """边界部分 _第一层"""
    with tf.name_scope("Edge_part"):
        with tf.name_scope("Edge_layer1"):
            E10=a10
            E10=relu_(BN(conv_1x1(inputs=E10,filters=numclasses),training=is_training))
            E11=a11
            E11=relu_(BN(conv_1x1(inputs=E11,filters=numclasses),training=is_training))
        """第二层"""
        with tf.name_scope("Edge_layer2"):
            E20=a20
            E20=relu_(BN(conv_1x1(inputs=E20,filters=numclasses),training=is_training))
            E21=a21
            E21=relu_(BN(conv_1x1(inputs=E21,filters=numclasses),training=is_training))
        """ 第三层"""
        with tf.name_scope("Edge_layer3"):
            E30=a30
            E30=relu_(BN(conv_1x1(inputs=E30,filters=numclasses),training=is_training))
            E31=a31
            E31=relu_(BN(conv_1x1(inputs=E31,filters=numclasses),training=is_training))
            E32=a32
            E32=relu_(BN(conv_1x1(inputs=E32,filters=numclasses),training=is_training))
        """第四层"""
        with tf.name_scope("Edge_layer4"):
            E40=a40
            E40=relu_(BN(conv_1x1(inputs=E40,filters=numclasses),training=is_training))
            E41=a41
            E41=relu_(BN(conv_1x1(inputs=E41,filters=numclasses),training=is_training))
            E42=a42
            E42=relu_(BN(conv_1x1(inputs=E42,filters=numclasses),training=is_training))
        """第五层"""
        with tf.name_scope("Edge_layer5"):
            E50=a50
            E50=relu_(BN(conv_1x1(inputs=E50,filters=numclasses),training=is_training))
            E51=a51
            E51=relu_(BN(conv_1x1(inputs=E51,filters=numclasses),training=is_training))
            E52=a52
            E52=relu_(BN(conv_1x1(inputs=E52,filters=numclasses),training=is_training))

        Edge_merge=[]   # 将边界部分考虑进解码端

        """融合阶段"""
        with tf.name_scope("Edge_Merge"):
            FE1=tf.concat([E10, E11], axis=3)
            FE1=relu_(BN(conv_1x1(inputs=FE1,filters=2),training=is_training))

            Edge_merge.append(FE1) # 将边界部分考虑进解码端

            FE2=tf.concat([E20,E21], axis=3)
            FE2=relu_(BN(conv_1x1(inputs=FE2,filters=2),training=is_training))

            Edge_merge.append(FE2) # 将边界部分考虑进解码端


            FE3=tf.concat([E30,E31,E32], axis=3)
            FE3=relu_(BN(conv_1x1(inputs=FE3,filters=2),training=is_training))

            Edge_merge.append(FE3)# 将边界部分考虑进解码端


            FE4=tf.concat([E40,E41,E42], axis=3)
            FE4=relu_(BN(conv_1x1(inputs=FE4,filters=2),training=is_training))

            Edge_merge.append(FE4) # 将边界部分考虑进解码端

            FE5=tf.concat([E50,E51,E52], axis=3)
            FE5=relu_(BN(conv_1x1(inputs=FE5,filters=2),training=is_training))

            Edge_merge.append(FE5)  # 将边界部分考虑进解码端

        Edge_merge=reversed(Edge_merge[:-1]) # 将边界部分考虑进解码端

        """上采样"""
        with tf.name_scope("Up_Sampling"):
            F21=relu_(BN(conv_transpose(inputs=FE2,strides=(2,2)),training=is_training))
            F31=relu_(BN(conv_transpose(inputs=FE3,strides=(4,4)),training=is_training))
            F41=relu_(BN(conv_transpose(inputs=FE4,strides=(8,8)),training=is_training))
            F51=relu_(BN(conv_transpose(inputs=FE5,strides=(16,16)),training=is_training))

            edge=tf.concat([FE1,F21,F31,F41,F51],axis=3)
            edgeresult=conv_1x1(inputs=edge,filters=2)

    """解码端"""
    with tf.name_scope("Encoder"):
        convs = reversed(convs[:-1])
        input = a52
        for index, c in enumerate(reversed(_CHANNELS[:-1])):
            if c<_CHANNELS[2]:
                _DEPTH = 2
            else:
                _DEPTH = 3
            itet_conv=next(convs)
            deconv_shape=input.get_shape()
            W_t1=weight_variable([4,4,c,deconv_shape[-1]], name="W_t1_"+str(index))
            b_t1=bias_variable([c], name="b_t1"+str(index))
            conv_t1=conv2d_transpose_strided(input, W_t1, b_t1, output_shape=tf.shape(itet_conv))
            input=BN(conv_t1,training=is_training)
            input=tf.concat([input, itet_conv,next(Edge_merge)], axis=3)  # 将边界部分考虑进解码端
            input=channel_block(input_feature=input,name='Channel_attention'+str(index),ratio=8)
            if index==0:
                input=Inception_block(input=input, is_training=is_training,filters={"1x1": 64, "3x3": [256, 128], "5x5": [64, 32], "7x7": [32, 32]}, name_layer="Inception_destage1")
                input=Inception_block(input=input, is_training=is_training,filters={"1x1": 64, "3x3": [256, 128], "5x5": [64, 32], "7x7": [32, 32]}, name_layer="Inception_destage2")
            else:
                for _ in range(_DEPTH-1):
                    input = relu_(BN(conv_3x3(inputs=input, filters=c), training=is_training))
        logits=conv_1x1(inputs=input, filters=numclasses)

    return logits,edgeresult


if __name__ == "__main__":
    inputs = tf.random.normal([1, 256, 256, 3])
    result,_= BEDSN(inputs, tf.constant(False,dtype=tf.bool))#False？
    print(result.get_shape().as_list())