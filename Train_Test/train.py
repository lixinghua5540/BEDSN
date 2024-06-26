# -*- coding: utf-8 -*-
import time
now_time = time.time()
import importlib
import logging
import os
import sys
#sys.path.append('./Semantic_Segmentation')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import warnings
warnings.filterwarnings("ignore")
import numpy as np

import tensorflow as tf
from tensorflow import ConfigProto###动态分配显存降低显存占用
from tensorflow import InteractiveSession
config=ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
print(f"GPU_Available:{tf.test.is_gpu_available()}")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)

from data import next_batch
from cau_time import get_time_dif
from Parameters_Set import *
print(f"所选网络为:{Name_select_network}")
if_training=tf.constant(True, dtype=tf.bool)
dir_name=Networks_dic[Name_select_network][0]   # 文件夹名称
Name_operation=Networks_dic[Name_select_network][1]  # 模型操作名称
Name_network='.'.join([root_name,dir_name,Name_select_network]) # 模型文件绝对地址【从父目录开始】，这里在部署的时候非常容易出错
module_name=importlib.import_module(Name_network)  # 获取模型文件,总之这个函数有问题,很难引用父目录中的内容
inference_model=getattr(module_name,Name_operation) # 导入模型中的操作函数
print(inference_model)

img=tf.placeholder(tf.float32, [batch_size, 256, 256, 3])   # 影像的大小，根据需要自行调整 float32？
label=tf.placeholder(tf.int32, [batch_size, 256, 256])#label 化成了1通道
phase_train = tf.placeholder(tf.bool, name='phase_train')#phase train的意义在哪里
#phase_train = tf.constant(True, dtype=tf.bool)
pred=inference_model(img,phase_train)
#pred=inference_model(img)

# update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)

"""" 需注意：当利用tf.keras.layers.BatchNormalization()实现 BN时,不会自动将 update_ops 添加到 tf.GraphKeys.UPDATE_OPS 这个 collection 中。因此需要手动添加~ """

update_ops1= tf.get_collection(tf.GraphKeys.UPDATE_OPS)
ops = tf.get_default_graph().get_operations()
bn_update_ops =[ x for x in ops if ("AssignMovingAvg" in x.name and x.type=="AssignSubVariableOp")]
tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,bn_update_ops)
update_ops_=tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # 两维 列表中包含列表元素
update_ops=update_ops_[:-1]

for v in range (len(update_ops_[-1])):
    update_ops.append(update_ops_[-1][v])

# file = open('./update_ops.txt', 'w')  # 参数提取
# for v in update_ops:
#     file.write(str(v.name)+'\n')
# file.close()


with tf.control_dependencies(update_ops):         # control_dependencies 是tensorflow中的一个flow顺序控制机制
    Loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))  # 计算logits和labels 之间的稀疏softmax 交叉熵，这里应该不太适合用整型数据
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(pred,axis=-1),tf.int32), label), tf.float32))
    train_step = tf.train.AdamOptimizer(learning_rate=3e-4).minimize(Loss)   #寻找全局最优点的优化算法

    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('cross_entropy_loss', Loss)


saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=10)  # 目标：训练网络后想保存训练好的模型，以及在程序中读取以保存的训练好的模型

def load():
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir=save_root+'checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))  #重载模型的参数，继续训练或用于测试数据
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        return True, counter

    else:
        print(" [*] Failed to find a checkpoint")
        return False, 0

def train():
    # losses=np.load(save_root+"loss.npy").tolist()
    # accuracies=np.load(save_root+"acc.npy").tolist()

    losses=[]
    accuracies=[]
    # l2_losses=[]

    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # 设置日志级别，低于级别的不再输出
    handler = logging.StreamHandler()  # output to standard output
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))  # 设置输出格式
    logger.addHandler(handler)

    tf.global_variables_initializer().run()
    could_load, checkpoint_counter = load()
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(save_root+'logs/')
    writer.add_graph(sess.graph)

    #print("could_load:  ",could_load," *****  ","checkpoint_counter:  ", checkpoint_counter,"!!!      ")
    if could_load:
        start_epoch = (int)(checkpoint_counter/num_batches)
        start_batch_id = checkpoint_counter-start_epoch*num_batches
        counter = checkpoint_counter
        #print("start_epoch: ",start_epoch,"    start_batch_id:  ",start_batch_id,"   counter:",counter)
        print(" [*] Load SUCCESS")
        print(" [*] Load SUCCESS")

    else:

        start_epoch = 0
        start_batch_id = 0
        counter=0
        print(" [!] Load failed...")
    for i in range(start_epoch,train_epoches):
        for j in range(start_batch_id,num_batches):
            end=time.time()
            x_batch, y_batch = next_batch()#a typical numpy array
            print(x_batch.shape)
            feed_dict = {img: tf.constant(np.float32(x_batch),dtype=tf.float32),#
                         label: tf.constant(np.int32(y_batch),dtype=tf.int32),
                         phase_train:if_training#就只有pred用到了phase_train
                         #phase_train:True
                         }
            #print("fd",sess.run(feed_dict))
            ts,s,loss, pred1,acc= sess.run([train_step,summ,Loss, pred,accuracy], feed_dict=sess.run(feed_dict))#用GPU的坏处

            batch_time = time.time() - end
            remain_iter = train_epoches * num_batches - (i * num_batches + j)
            remain_time = remain_iter * batch_time
            t_m, t_s = divmod(remain_time, 60)  # (x//y, x%y)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            logger.info('Epoch: [{}/{}][{}/{}]   '
                        'Loss: {loss:.6f}   '
                        'Accuracy: {acc:.5f}   '
                        #'Cross_Entropy_Loss: {CE_loss:.6f}  '
                        'Remain_time: {remain_time}'.format(i + 1, train_epoches, j + 1, num_batches,
                                                            loss=loss,
                                                            acc=acc,
                                                            #CE_loss=sess.run(Loss),
                                                            remain_time=remain_time
                                                            ))
            losses.append(loss)
            accuracies.append(acc)
            counter+=1
            writer.add_summary(s, counter)
        start_batch_id=0
        saver.save(sess, save_root+'checkpoint/'+Name_select_network+'.ckpt', global_step=counter,write_meta_graph=True if i==0 else False)

        # 训练循环中，定期调用 saver.save()，向文件夹中写入包含当前模型中所有可训练变量的 checkpointfew 文件
        # global_step表示迭代次数
        # 计算图随着训练的进行不会改变，write_meta_graph=True if i==0 else False用于仅保存依次计算图
        np.save(save_root+"loss.npy", losses)
        np.save(save_root+"acc.npy", accuracies)

with tf.Session() as sess:
    train()

total_time= get_time_dif(now_time)
print("total_time", total_time)  ##4##