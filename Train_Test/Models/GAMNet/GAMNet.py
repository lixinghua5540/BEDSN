# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from Sub_modules import  GAM, Atrous_spatial_pyramid_pooling
from Semantic_Segmentation.Common_Codes.Train_Test.Parameters_Set import if_training,numclasses


def CONV2D(input,filter):
    result=tf.layers.conv2d(inputs=input,
                             filters=filter,
                             kernel_size=(1,1),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    return result
def CONV3D(input,filter):
    result=tf.layers.conv2d(inputs=input,
                             filters=filter,
                             kernel_size=(3, 3),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    return result

# 一系列ResNet 网路的设计
"""基础结构，什么时候进行下采样是关键的一步"""
class BasicBlock(layers.Layer):
    expansion = 1    # 针对卷积核一致与否
    def __init__(self, out_channel, strides=1, downsample=None,atrous_rate=(1,1), **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,dilation_rate=atrous_rate,padding="SAME", use_bias=False)
        # padding="SAME"时，输出向上取整数
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,dilation_rate=atrous_rate,padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu=layers.ReLU()
        self.add=layers.Add()

    def __call__(self, inputs, training=if_training):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs,training=training)
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)    #  training 训练与测试时体现不同
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)
        return x

"""由inputs--->x，通过瓶颈捷径"""
class Bottleneck(layers.Layer):
    expansion = 4
    def __init__(self, out_channel, strides=1, downsample=None,atrous_rate=(1,1), **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False,dilation_rate=atrous_rate, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,dilation_rate=atrous_rate,strides=strides, padding="SAME", name="conv2")  # 步长取决于不同结构
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False,dilation_rate=atrous_rate,name="conv3")    # 卷积核的个数与第一层相比是其四倍
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample =downsample
        self.add=layers.Add()

    def __call__(self, inputs, training=if_training):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs,training=training)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x

def _make_layer(block, in_channel, channel, block_num, name, strides=1,dialted_rate=(1,1)):
       #  block basic net/battlenet'
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNormdownsample")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, atrous_rate=dialted_rate,name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, atrous_rate=dialted_rate,name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)

def _resnet(input,block, blocks_num,training):
    #include_top  顶层结构  全连接层XX

    convs=[]
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,padding="SAME", use_bias=False, name="conv1")(input)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x,training=training)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    convs.append(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    convs.append(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    convs.append(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], dialted_rate=(2,2), name="block4")(x)  # 最后一块换为空洞卷积
    convs.append(x)

    x=Atrous_spatial_pyramid_pooling(x,filters=x.shape[-1],A_training=training)

    x=GAM(x,convs[-1],x.shape[-1],is_training=training)
    x=_make_layer(BasicBlock, x.shape[-1], 512*2, 3,  name="block5")(x)  # 最后一块换为空洞卷积
    convs = reversed(convs[:-1])

    x=GAM(x,next(convs),x.shape[-1],is_training=training)
    x=_make_layer(BasicBlock, x.shape[-1], 256*2, 3,name="block6")(x)
    x= tf.image.resize(x, tf.shape(x)[1:3] * 2)
    x = GAM(x, next(convs), x.shape[-1], is_training=training)
    x = _make_layer(BasicBlock, x.shape[-1], 128*2,3,name="block7")(x)
    x = tf.image.resize(x, tf.shape(x)[1:3] * 2)
    x = GAM(x, next(convs), x.shape[-1], is_training=training)
    x = _make_layer(BasicBlock, x.shape[-1], 64*2, 3, name="block8")(x)

    x = tf.layers.conv2d_transpose(x, filters=64,
                                   kernel_size=(4, 4),
                                   strides=(4, 4),
                                   padding='valid',
                                   activation=None)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="de-conv")(x,training=training)
    x = tf.nn.relu(x)

    logits = tf.layers.conv2d(inputs=x,
                              filters=numclasses,
                              kernel_size=(1, 1),
                              kernel_initializer=tf.variance_scaling_initializer(),
                              padding='SAME', activation=None)

    return logits

def GAMNet_resnet101(input,training=False):
    result=_resnet(input,Bottleneck, [3, 4, 23, 3],training=training)
    return result

# if __name__ == "__main__":
#     inputs = tf.random.normal([1, 256, 256, 3])
#     result = GAMNet_resnet101(inputs, tf.constant(False,dtype=tf.bool))
#     print(result.get_shape().as_list())

