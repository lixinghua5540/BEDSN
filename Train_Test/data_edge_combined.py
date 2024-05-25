# -*- coding: utf-8 -*-
"""对应于：联合边缘检测的语义分割算法，此部分代码可直接利用Tensorflow框架进行数据的分批获取"""

import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")
from  skimage import  io
from Parameters_Set import  batch_size

def load_batch(x, y,z):
    x1 = []
    y1 = []
    z1=[]
    for i in range(len(x)):
        img = io.imread(x[i]) / 255.0
        lab = io.imread(y[i])
        edg_lab=io.imread(z[i])
        x1.append(img)
        y1.append(lab)
        z1.append(edg_lab)
    x1=np.array(x1)
    y1=np.array(y1)
    z1=np.array(z1)
    return x1, y1,z1

def next_batch():
    img = np.array(sorted(glob.glob(r'./Datasets/Vaihingen/train_images/*.png')))  # 影像保存路径，值得注意的是这里应该是在cpu上处理
    label = np.array(sorted(glob.glob(r'./Datasets/Vaihingen/train_newlabels/*.png')))  #语义分割类别标签保存路径
    edge_label=np.array(sorted(glob.glob(r'./Datasets/Vaihingen/train_edgelabels/*.png'))) # 边缘检测类别标签保存路径

    index=np.random.choice(len(img),batch_size)
    x_batch=img[index]
    y_batch=label[index]
    z_batch=edge_label[index]
    image_batch, lab_batch,edge_batch=load_batch(x_batch, y_batch,z_batch)
    return image_batch, lab_batch,edge_batch

def next_val_batch():
    img=np.array(sorted(glob.glob(r'E:/val_images/*.png'))) # 验证集影像保存路径
    label=np.array(sorted(glob.glob(r'E:/val_labels/*.png'))) # 验证集语义分割类别标签保存路径
    edge_label=np.array(sorted(glob.glob(r"E:/val_edgelabels/*.png")))# 验证集边缘检测类别标签保存路径

    index=np.random.choice(len(img),batch_size)
    x_batch=img[index]
    y_batch=label[index]
    z_batch=edge_label[index]
    image_batch, lab_batch,edge_batch=load_batch(x_batch, y_batch,z_batch)
    return image_batch, lab_batch,edge_batch