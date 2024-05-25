# -*- coding: utf-8 -*-
"""使用边界标签或膨胀的边界标签计算预测结果边界标签的准确程度"""
import  time
start_time=time.time()
import  numpy as np
import  glob
from PIL import Image
import pandas as pd
from collections import Counter
from skimage import  io
import  os
import tensorflow as tf
from datetime import timedelta
import sys
import tensorflow as tf
sys.path.append("./Semantic_Segmentation/Common_Codes/Train_Test")
from Parameters_Set import *
def convert_edge2RGB(save_root,edgeresults_path):
    for i in range(len(edgeresults_path)):#遍历每个影像
        save_path_edgeRGB =save_root+"Plurality_Voting/Edge_DRGB/"#有问题
        if not os.path.exists(save_path_edgeRGB):
            os.makedirs(save_path_edgeRGB)
        results = io.imread(edgeresults_path[i])#uint8 单通道
        label_tf=tf.convert_to_tensor(results)#
        label_tf=tf.expand_dims(tf.expand_dims(label_tf,0),-1)#
        D_label=tf.layers.max_pooling2d(label_tf,pool_size=[3,3],strides=[1,1],padding='SAME',data_format='channels_last')#
        session = tf.Session()#
        D_label=np.squeeze(session.run(D_label))#
        results=D_label#
        #创建三通道影像
        R_edge=np.zeros_like(results)
        G_edge=np.zeros_like(results)
        B_edge=np.zeros_like(results)
        inds=(results==1)
        R_edge[inds]=255
        G_edge[inds]=255
        B_edge[inds]=255
        R_edge=np.expand_dims(R_edge,axis=-1)
        G_edge=np.expand_dims(G_edge,axis=-1)
        B_edge=np.expand_dims(B_edge,axis=-1)
        RGB_edge=np.concatenate((R_edge,G_edge,B_edge),axis=-1)
        name_ = os.path.basename(edgeresults_path[i]).split(".")[0]
        print(name_)
        image_edge=Image.fromarray(RGB_edge)
        image_edge.save(save_path_edgeRGB+name_+'.tif')
        #print(RGB_edge.shape)
if __name__ == "__main__": 
    #dataset='Vaihingen'
    #datasource='./Semantic_Segmentation/Results/'+dataset+'/'
    #net_list=os.listdir(datasource)
    #print(net_list)
    #for i in range(len(net_list)):
    #    if i!=1:
    #        save_root= datasource+net_list[i]+'/'
    #        #path_GT_edge_labels='./Datasets/'+dataset+'/test_newedges/'
    #        edgeresults_path=np.array(sorted(glob.glob(save_root+"Plurality_Voting/Edge/*.tif"))) # 预测结果
    #        convert_edge2RGB(save_root,edgeresults_path)
    save_root= './Datasets/Vaihingen/'
    #path_GT_edge_labels='./Datasets/'+dataset+'/test_newedges/'
    edgeresults_path=np.array(sorted(glob.glob(save_root+"test_newedges/*.tif"))) # 预测结果
    convert_edge2RGB(save_root,edgeresults_path)