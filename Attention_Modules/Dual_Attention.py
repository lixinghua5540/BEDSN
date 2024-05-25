# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from tensorflow.layers import conv2d
from tensorflow.nn import softmax


# 卷积层中的卷积函数
def Conv_1(input,filter):
    result=tf.layers.conv2d(inputs=input,
                             filters=filter,
                             kernel_size=(1, 1),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    return result

def Conv_3(input,filter):
    result=tf.layers.conv2d(inputs=input,
                             filters=filter,
                             kernel_size=(3, 3),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    return result

def Conv_BN_relu(input,filter,is_training):
    output=Conv_3(input,filter)
    output=tf.layers.batch_normalization(inputs=output,
                                          training=is_training,
                                          center=True,
                                          scale=True,
                                          fused=True)
    output=tf.nn.relu(output)
    return output


# 上采样中采用的卷积函数
def PAM_Module(input_feature):
    with tf.name_scope("PAM_Module"):
        Batch_size,Height,Width,Channels=input_feature.shape #[Batch_size,Height,Width,Channels]
        ratio=8
        lamda=tf.Variable(initial_value=0,trainable=True,name='parameter_PAM')

        proj_Query=Conv_1(input_feature,Channels//ratio)    # [Batch_size,Height,Width,Channels(//ratio)]
        proj_Query=tf.reshape(proj_Query,[Batch_size,Height*Width,-1])   # [Batch_size,Height*Width,Channels(//ratio)]

        proj_Key=Conv_1(input_feature, Channels//ratio)    # [Batch_size,Height,Width,Channels(//ratio)]
        proj_Key=tf.transpose(tf.reshape(proj_Key,[Batch_size,Height*Width,-1]),perm=[0,2,1] ) # [Batch_size,Channels(//ratio),Height*Width]
        Energy=tf.matmul(proj_Query,proj_Key) # [Batch_size,Height*Width,Height*Width]
        Attention=tf.nn.softmax(Energy)

        proj_Value = Conv_1(input_feature, Channels)  # [Batch_size,Height,Width,Channels]
        proj_Value=tf.reshape(proj_Value,[Batch_size,Height*Width,-1]) # [Batch_size,Height*Width,Channels]

        out=tf.matmul(tf.transpose(Attention,perm=[0,2,1]),proj_Value) # [Batch_size,Height*Width,Channels]
        out=tf.reshape(out,[Batch_size,Height,Width,-1]) #[Batch_size,Height,Width,Channels]
        out=lamda*out+input_feature
    return out

def CAM_Module(input_feature):
    with tf.name_scope("CAM_Module"):
        Batch_size,Height,Width,Channels_=input_feature.shape  # [Batch_size,Height,Width,Channels]
        beta=tf.Variable(initial_value=0, trainable=True, name='parameter_PAM')

        proj_Query = tf.reshape(input_feature,[Batch_size,Height*Width,-1])  # [Batch_size,Height*Width,Channels]
        proj_Key=tf.transpose(proj_Query,perm=[0,2,1]) # [Batch_size,Channels,Height*Width]
        Energy=tf.matmul(proj_Key,proj_Query)   # [Batch_size,Channels,Channels]
        # Energy_new=tf.tile(tf.reduce_max(Energy,keepdims=True)[0],[1,Energy.shape[1],Energy.shape[2],Energy.shape[3]])-Energy
        Attention=tf.nn.softmax(Energy)

        proj_Value=tf.reshape(input_feature,[Batch_size,Height*Width,-1])  # [Batch_size,Height*Width,Channels]
        out=tf.matmul(proj_Value,tf.transpose(Attention, perm=[0, 2, 1]))  # [Batch_size,Height*Width,Channels]
        out=tf.reshape(out, [Batch_size, Height, Width, -1])#[Batch_size, Height, Width, Channels]
        out=beta*out+input_feature
    return out

def DualAttentionMoudle(input_feature):
    with tf.name_scope("Dual_Attention"):
        PAM_feature=PAM_Module(input_feature)
        CAM_feature=CAM_Module(input_feature)
        DAMfeature=PAM_feature+CAM_feature
    return DAMfeature
