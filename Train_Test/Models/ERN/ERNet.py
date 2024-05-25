# -*- coding: utf-8 -*-
import numpy as np
import os
import tensorflow as tf
from Semantic_Segmentation.Common_Codes.Train_Test.Parameters_Set import numclasses

def BN(inputs,training):
    return tf.layers.batch_normalization(inputs=inputs,training=training,center=True,scale=True,fused=True)


def conv_1x1(inputs,filters):
    return tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(1,1),
                            kernel_initializer=tf.variance_scaling_initializer(),use_bias=False,
                            padding='SAME', activation=None)

def relu_(input):
    return tf.nn.relu(input)

def maxpool_2x2(inputs,name_layer):
    return tf.layers.max_pooling2d(inputs=inputs,pool_size=(2, 2),strides=(2, 2),name=name_layer+"_maxpool")

def conv_transpose(inputs,filter,is_training,kesize,name_layer):
    output=tf.layers.conv2d_transpose(inputs=inputs,filters=filter,kernel_size=kesize,
                                      kernel_initializer=tf.variance_scaling_initializer(),use_bias=False,
                                      strides=kesize, padding='valid',activation=None,name=name_layer+"_deconv")
    output=tf.layers.batch_normalization(inputs=output,training=is_training,name=name_layer+'_BN')
    output=tf.nn.relu(output,name=name_layer+'_relu')
    return output


def CONV_BN_ReLU(input,filter,kesize,stride,is_training,name_layer):
    input=tf.layers.conv2d(inputs=input,filters=filter,kernel_size=kesize,strides=stride,padding="SAME",use_bias=False,
                           kernel_initializer=tf.variance_scaling_initializer(),name=name_layer+"_conv")
    input=tf.layers.batch_normalization(inputs=input,training=is_training,name=name_layer+'_BN')
    input=tf.nn.relu(input,name=name_layer+'_relu')
    return input

def Residual_Block(input,is_training,filters,name_layer):
    filters_name=list(filters.keys())
    assert filters_name==["1x1",'Bottle'],"NameError in Residual_Block module."
    Branch_1x1=BN(conv_1x1(input,filters=filters['1x1']),training=is_training)
    Branch_HS=CONV_BN_ReLU(input=input,filter=filters['Bottle'][0],kesize=1,stride=1,is_training=is_training,name_layer=name_layer+"_conv1x1")
    Branch_HS=CONV_BN_ReLU(input=Branch_HS,filter=filters['Bottle'][1],kesize=3,stride=1,is_training=is_training,name_layer=name_layer+"Bottle")
    Branch_HS=BN(conv_1x1(Branch_HS,filters=filters['Bottle'][2]),training=is_training)
    output=tf.add(Branch_HS,Branch_1x1,name=name_layer+'_add')
    output=relu_(output)
    return output

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


def ERNet(input_images,is_training=True):
    Channels=[64,128,256,512]
    Edge_Encode=[]  # 语义分割模型编码端构成的边界监督构成
    Edge_Decode=[] # 语义分割模型解码端构成的边界监督构成

    with tf.name_scope("Block_A"):  # 256x256 @64
        input=CONV_BN_ReLU(input_images,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Layer1_stage1")
        input=CONV_BN_ReLU(input,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Layer1_stage2")
    input=maxpool_2x2(input,name_layer='Downsample1')

    with tf.name_scope("Block_B"): # 128x128 @128
        input=CONV_BN_ReLU(input,filter=Channels[1],kesize=3,stride=1,is_training=is_training,name_layer="Layer2_stage1")
        input=CONV_BN_ReLU(input,filter=Channels[1],kesize=3,stride=1,is_training=is_training,name_layer="Layer2_stage2")
        Edge_Encode.append(input)

    with tf.name_scope("Residual_B_r"): # 128x128 @128
        residual_input=input
        residual_input=Residual_Block(residual_input,is_training=is_training,filters={"1x1":128,'Bottle':[64,128,128]},name_layer="Residual1")
    input=maxpool_2x2(input,name_layer='Downsample2')

    with tf.name_scope("Inception_C"): # 64x64 @256
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[128,128],"5x5":[32,32],"7x7":[32,32]},name_layer="Layer3_stage1")
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[128,128],"5x5":[32,32],"7x7":[32,32]},name_layer="Layer3_stage2")
        assert input.get_shape().as_list()[-1]==Channels[-2],"ChannelMatch Error in Inception_D."
        Edge_Encode.append(input)

    with tf.name_scope("Residual_C_r"): # 64x64 @256
        residual_input1=input
        residual_input1=Residual_Block(residual_input1,is_training=is_training,filters={"1x1":256,'Bottle':[64,128,256]},name_layer="Residual2")

    input=maxpool_2x2(input,name_layer='Downsample3')

    with tf.name_scope("Inception_D"):  #
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[192,256],"5x5":[64,128],"7x7":[32,64]},name_layer="Layer4_stage1")
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[192,256],"5x5":[64,128],"7x7":[32,64]},name_layer="Layer4_stage2")
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[192,256],"5x5":[64,128],"7x7":[32,64]},name_layer="Layer4_stage3")
        assert input.get_shape().as_list()[-1]==Channels[-1],"ChannelMatch Error in Inception_D."

    with tf.name_scope("Deconvolution_E"):
        input=conv_transpose(inputs=input,filter=Channels[2],is_training=is_training,kesize=2,name_layer="Layer5_Upsample")
        input=tf.concat([input,residual_input1],axis=-1)

    with tf.name_scope("Inception_F"):
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[256,128],"5x5":[64,32],"7x7":[32,32]},name_layer="Layer6_stage1")
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[256,128],"5x5":[64,32],"7x7":[32,32]},name_layer="Layer6_stage2")
        input=Inception_block(input=input,is_training=is_training,filters={"1x1":64,"3x3":[256,128],"5x5":[64,32],"7x7":[32,32]},name_layer="Layer6_stage3")
        Edge_Decode.append(input)

    with tf.name_scope("Deconvolution_G"):
        input=conv_transpose(inputs=input,filter=Channels[1],is_training=is_training,kesize=2,name_layer="Layer7_Upsample")
        input=tf.concat([input,residual_input],axis=-1)

    with tf.name_scope("Block_H"):
        input=CONV_BN_ReLU(input=input,filter=Channels[1],kesize=3,stride=1,is_training=is_training,name_layer="Layer8_stage1")
        input=CONV_BN_ReLU(input=input,filter=Channels[1],kesize=3,stride=1,is_training=is_training,name_layer="Layer8_stage2")
        Edge_Decode.append(input)

    with tf.name_scope("Deconvolution_S"):
        logits=tf.layers.conv2d_transpose(inputs=input, filters=numclasses, kernel_size=2,
                               kernel_initializer=tf.variance_scaling_initializer(), use_bias=True,
                               strides=2, padding='valid', activation=None, name="Layer9_Upsample_deconv")

    with tf.name_scope("Edge_Part"):
        Encode_edge1=conv_transpose(inputs=Edge_Encode[0],filter=2,is_training=is_training,kesize=2,name_layer="Edge_Encode_Be")
        Encode_edge2=conv_transpose(inputs=Edge_Encode[1],filter=2,is_training=is_training,kesize=4,name_layer="Edge_Encode_Ce")
        Edge_encode=tf.concat([Encode_edge1,Encode_edge2],axis=-1)
        with tf.name_scope("Edge_Conv_Encode"):
            Edge_encode=CONV_BN_ReLU(Edge_encode,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Edge_Encode_conv1")
            Edge_encode=CONV_BN_ReLU(Edge_encode,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Edge_Encode_conv2")
            EEL=tf.layers.conv2d(inputs=Edge_encode,filters=2,kernel_size=3,padding="SAME",kernel_initializer=tf.variance_scaling_initializer(),name="EEL") #编码端边界监督的输出

        Deconde_edge1=conv_transpose(inputs=Edge_Decode[0],filter=2,is_training=is_training,kesize=4,name_layer="Edge_Decode_Fe")
        Deconde_edge2=conv_transpose(inputs=Edge_Decode[1],filter=2,is_training=is_training,kesize=2,name_layer="Edge_Decode_He")
        Edge_decode=tf.concat([Deconde_edge1,Deconde_edge2],axis=-1)
        with tf.name_scope("Edge_Conv_Decode"):
            Edge_decode=CONV_BN_ReLU(Edge_decode,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Edge_Decode_conv1")
            Edge_decode=CONV_BN_ReLU(Edge_decode,filter=Channels[0],kesize=3,stride=1,is_training=is_training,name_layer="Edge_Decode_conv2")
            DEL=tf.layers.conv2d(inputs=Edge_decode,filters=2,kernel_size=3,padding="SAME",kernel_initializer=tf.variance_scaling_initializer(),name="DEL") #解码端边界监督的输出
    return logits,[EEL,DEL]   # 在调用该算法时，注意输出赋值变量的格式



# if __name__ == "__main__":
#     inputs = tf.random.normal([1, 256, 256, 3])
#     result,_= ERNet(inputs, tf.constant(False,dtype=tf.bool))
#     print(result.get_shape().as_list())