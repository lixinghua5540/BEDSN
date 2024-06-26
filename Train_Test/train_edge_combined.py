# -*- coding: utf-8 -*-
import time
now_time = time.time()
import importlib
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import sys 
sys.path.append("Semantic_Segmentation/Common_Codes") 
import warnings
warnings.filterwarnings("ignore")
import numpy as  np

import tensorflow as tf
print(f"GPU_Available:{tf.test.is_gpu_available()}")
from tensorflow import ConfigProto###动态分配显存降低显存占用
from tensorflow import InteractiveSession
config=ConfigProto()
config.gpu_options.allow_growth=True
session=InteractiveSession(config=config)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from Losses.Losses import loss_edge#引用父目录
from cau_time import get_time_dif
from data_edge_combined import next_batch
from Parameters_Set import *
print(f"所选网络为:{Name_select_network}")

dir_name=Networks_dic[Name_select_network][0]   # 文件夹名称
Name_operation=Networks_dic[Name_select_network][1]  # 模型操作名称
Name_network='.'.join([root_name,dir_name,Name_select_network]) # 模型文件绝对地址【从父目录开始】
module_name=importlib.import_module(Name_network)  # 获取模型文件
inference_model=getattr(module_name,Name_operation) # 导入模型中的操作函数

img=tf.placeholder(tf.float32, [batch_size, 256, 256, 3])
Seglabel=tf.placeholder(tf.int32, [batch_size, 256, 256])
Edgelabel=tf.placeholder(tf.int32, [batch_size, 256, 256])
phase_train = tf.placeholder(tf.bool, name='phase_train')
Segpred,Edgepred=inference_model(img,phase_train)# 该处training控制一部分，bottleneck中的training控制resnet50残差结构中的104个

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
    SegCEloss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Seglabel,logits=Segpred))  # 计算logits和labels 之间的稀疏softmax 交叉熵
    EdgeCEloss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Edgelabel,logits=Edgepred))
    lamda=0.5  #
    Edgeloss=loss_edge(labels=Edgelabel, logits=Edgepred)
    accuracy_seg=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(Segpred, axis=-1), tf.int32), Seglabel), tf.float32))
    accuracy_edge=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(Edgepred, axis=-1), tf.int32), Edgelabel), tf.float32))

    """
    EdgeCEloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Edgelabel, logits=Edgepred[1])) + 
                               tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Edgelabel, logits=Edgepred[0]))
    lamda=0.25
    Edgeloss=loss_edge(labels=Edgelabel, logits=Edgepred[0]) + loss_edge(labels=Edgelabel, logits=Edgepred[1])
    accuracy_seg=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(Segpred, axis=-1), tf.int32), Seglabel), tf.float32))
    accuracy_edg=tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(Edgepred[1], axis=-1), tf.int32), Edgelabel), tf.float32))
      """ # ERN

    Loss=tf.add(SegCEloss,tf.multiply(lamda,Edgeloss))
    train_step=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(Loss)  # 寻找全局最优点的优化算法

    tf.summary.scalar('accuracy_seg', accuracy_seg)
    tf.summary.scalar('accuracy_edge', accuracy_edge)
    tf.summary.scalar('loss', Loss)
    tf.summary.scalar('SegCEloss', SegCEloss)
    tf.summary.scalar('Edge_CEloss',EdgeCEloss)
    tf.summary.scalar('Edgeloss', Edgeloss)

saver = tf.train.Saver(var_list=tf.global_variables(),max_to_keep=10)#只保存最近十个epoch

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
    # accuracies_seg=np.load(save_root+"acc_seg.npy").tolist()
    # accuracies_edge=np.load(save_root+"acc_edge.npy").tolist()
    # losses_edge=np.load(save_root+"loss_edge.npy").tolist()
    # losses_seg=np.load(save_root+"loss_seg.npy").tolist()

    losses=[]
    accuracies_seg=[]
    accuracies_edge=[]
    losses_edge = []
    losses_seg=[]

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

    if could_load:
        start_epoch = (int)(checkpoint_counter/num_batches)
        start_batch_id = checkpoint_counter-start_epoch*num_batches
        counter = checkpoint_counter
        print(" [*] Load SUCCESS")
        print(" [*] Load SUCCESS")

    else:
        start_epoch = 0
        start_batch_id = 0
        counter=0
        print(" [!] Load failed...")
    sess.graph.finalize()

    for i in range(start_epoch,train_epoches):
        for j in range(start_batch_id,num_batches):
            end=time.time()
            x_batch, y1_batch, y2_batch = next_batch()
            feed_dict = {img: x_batch,
                         Seglabel: y1_batch,
                         Edgelabel: y2_batch,
                         phase_train: if_training
                         }
            ts,s,loss,pred1,pred2,acc_seg,acc_edg,seg_loss,edge_loss,edge_celoss= sess.run([train_step,summ,Loss, Segpred,Edgepred,accuracy_seg,accuracy_edge,SegCEloss,Edgeloss,EdgeCEloss], feed_dict=feed_dict)
            batch_time = time.time() - end
            remain_iter = train_epoches*num_batches-(i * num_batches + j)
            remain_time = remain_iter*batch_time
            t_m, t_s = divmod(remain_time, 60)  # (x//y, x%y)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
            logger.info('Epoch: [{}/{}][{}/{}]   '
                        'Loss: {loss:.6f}   '
                        'Seg_loss: {seg_loss:.6f}  '
                        'Edge_loss: {edge_loss:.6f}  '
                        'Edge_CEloss :{edge_celoss:.6f}  '
                        'Acc_seg: {acc_seg:.5f}   '
                        'Acc_edge:{acc_edg:.5f}  '
                        'Remain_time: {remain_time}'.format(i + 1, train_epoches, j + 1, num_batches,
                                                            loss=loss,
                                                            seg_loss=seg_loss,
                                                            edge_loss=edge_loss,
                                                            edge_celoss=edge_celoss,
                                                            acc_seg=acc_seg,
                                                            acc_edg=acc_edg,
                                                            remain_time=remain_time
                                                            ))

            losses.append(loss)
            accuracies_seg.append(acc_seg)
            accuracies_edge.append(acc_edg)
            losses_edge.append(edge_loss)
            losses_seg.append(seg_loss)

            counter+=1
            writer.add_summary(s, counter)
        start_batch_id = 0
        saver.save(sess,save_root+'checkpoint/'+Name_select_network+'.ckpt',global_step=counter, write_meta_graph=True if i==0 else False)
        np.save(save_root+"loss.npy", losses)
        np.save(save_root+"acc_seg.npy", accuracies_seg)
        np.save(save_root+"acc_edge.npy", accuracies_edge)
        np.save(save_root+"loss_edge.npy", losses_edge)
        np.save(save_root+"loss_seg.npy", losses_seg)

with tf.Session() as sess:
    train()

total_time= get_time_dif(now_time)
print("total_time", total_time)  ##4##