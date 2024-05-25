# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

def Conv_1(input,filter):
    result=tf.layers.conv2d(inputs=input,
                             filters=filter,
                             kernel_size=(1, 1),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    return result

def INF(B,H,W):

    # a=tf.convert_to_tensor(float("inf"))
    # a1=tf.repeat(a,H)          #参数2为标量，将tenso进行扩增后转化为1维
    # a2=tf.diag(a1)
    # a3=tf.expand_dims(a2,0)
    # result=-tf.tile(a3,[B*W,1,1])  # tf.tile针对多维tensor 进行扩增
    # result2=-tf.tile(tf.expand_dims(tf.diag(tf.repeat(tf.convert_to_tensor(float("inf")),H)),0),[B*W,1,1])
    # assert result.shape==result2.shape,'出错了'

    return -tf.tile(tf.expand_dims(tf.diag(tf.repeat(tf.convert_to_tensor(float("inf")),H)),0),[B*W,1,1])

def CC_Attentiion(inputs):

    inputs_shape=tf.shape(inputs)
    Batch_size,Height,Width,Channels_=inputs_shape[0],inputs_shape[1],inputs_shape[2],inputs_shape[3]
    ratio=8
    lamda=tf.Variable(initial_value=1.0,trainable=True,name='parameter_attention')
    Channels=inputs.get_shape().as_list()[-1]

    """Affinity操作"""
    proj_Query=Conv_1(inputs,Channels//ratio)    # [Batch_size,Height,Width,Channels(//ratio)]
    proj_Query_H=tf.reshape(tf.transpose(proj_Query,perm=[0,2,1,3]),[Batch_size*Width,Height,-1])  # [Batch_size*Width,Height,Channels(//ratio)]
    proj_Query_W=tf.reshape(proj_Query,[Batch_size*Height,Width,-1])                               # [Batch_size*Height,Width,Channels(//ratio)]


    proj_Key=Conv_1(inputs, Channels//ratio)    # [Batch_size,Height,Width,Channels(//ratio)]
    proj_Key_H=tf.reshape(tf.transpose(proj_Key,perm=[0,2,3,1]),[Batch_size*Width,-1,Height])  #[Batch_size*Width,Channels(//ratio),Height]
    proj_Key_W=tf.transpose(tf.reshape(proj_Key,[Batch_size*Height,Width,-1]),perm=[0,2,1])    #[Batch_size*Height,Channels(//ratio),Width]


    Energy_H=tf.transpose(tf.reshape((tf.matmul(proj_Query_H,proj_Key_H)+INF(Batch_size,Height,Width)),[Batch_size,Width,Height,Height]),perm=[0,2,1,3]) # [Batch_size,Height,Width,Height]
                                       #||--------[Batch_size*Width,Height,Height]------------------||

    Energy_W=tf.reshape(tf.matmul(proj_Query_W,proj_Key_W),[Batch_size,Height,Width,Width])   #[Batch_size,Height,Width,Width]
            #||------[Batch_size*Height,Width,Width]----||
    # assert Energy_W.get_shape().as_list() == [Batch_size,Height,Width,Width], 'Error:Energy_H 获取错误!'

    Affi_Concate=tf.nn.softmax(tf.concat([Energy_H,Energy_W],axis=-1),axis=-1)
    # assert Affi_Concate.get_shape().as_list() == [Batch_size,Height,Width,Height+Width], 'Error:Affi_Concate 获取错误!'

    """Aggregation操作"""
    proj_Value = Conv_1(inputs, Channels)  # [Batch_size,Height,Width,Channels]
    proj_Value_H = tf.reshape(tf.transpose(proj_Value, perm=[0, 2, 3, 1]),[Batch_size * Width, -1, Height])  # [Batch_size*Width,Channels,Height]
    proj_Value_W = tf.transpose(tf.reshape(proj_Value, [Batch_size * Height, Width, -1]),perm=[0, 2, 1])  # [Batch_size*Height,Channels,Width]
    # assert proj_Value_H.get_shape().as_list() == [Batch_size * Width, Channels, Height], 'Error:proj_Value_H 获取错误!'
    # assert proj_Value_W.get_shape().as_list() == [Batch_size * Height, Channels, Width], 'Error:proj_Value_W 获取错误!'

    Att_H=tf.reshape(tf.transpose(Affi_Concate[:,:,:,0:Height],perm=[0,2,1,3]),[Batch_size*Width,Height,Height])  # [Batch_size*Width,Height,Height],
                                 #[Batch_size,Height,Width,Height]
                     #|---------- [Batch_size,Width,Height,Height]-------------|

    Att_W=tf.reshape(Affi_Concate[:,:,:,Height:Height+Width],[Batch_size*Height,Width,Width])#[Batch_size*Height,Width,Width],
                    # [Batch_size,Height,Width,Width]

    Out_H=tf.transpose(tf.reshape(tf.matmul(proj_Value_H,tf.transpose(Att_H,perm=[0,2,1])),[Batch_size,Width,-1,Height]),perm=[0,3,1,2,]) # [Batch_size,Height,Width,Channels]
                                   #|------- [Batch_size*Width,Channels,Height]----------|
                       # |----------------------------[Batch_size,Width,Channels,Height]------------------------------|

    Out_W=tf.transpose(tf.reshape(tf.matmul(proj_Value_W,tf.transpose(Att_W,perm=[0,2,1])),[Batch_size,Height,-1,Width]),perm=[0,1,3,2])  # [Batch_size,Height,Width,Channels]
                                  # |------- [Batch_size*Height,Channels,Width]----------|
                      # |----------------------------[Batch_size,Height,Channels,Width]------------------------------|
    # assert Out_H.get_shape().as_list()==[Batch_size,Height,Width,Channels], 'Error:Out_H 获取错误!'
    # assert Out_W.get_shape().as_list()==[Batch_size,Height,Width,Channels], 'Error:Out_W 获取错误!'

    OutPut=lamda*(Out_H+Out_W)+inputs
    # assert OutPut.get_shape().as_list()==[Batch_size,Height,Width,Channels], 'Error:OutPut 获取错误!'

    return OutPut
