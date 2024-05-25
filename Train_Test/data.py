# -*- coding: utf-8 -*-
"""对应于：不联合边缘检测的语义分割算法"""
import numpy as np
import glob
import warnings
warnings.filterwarnings("ignore")
from  skimage import  io
from Parameters_Set import  batch_size
def load_batch(x, y):
    x1 = []
    y1 = []
    for i in range(len(x)):
        img = io.imread(x[i]) / 255.0
        lab = io.imread(y[i])
        x1.append(img)
        y1.append(lab)
    x1 = np.array(x1)
    y1 = np.array(y1)
    #print(x1.dtype)
    #print(y1.dtype)
    #print(np.float32(x1).dtype)
    #print(np.int32(y1).dtype)
    return x1, y1#修改数据格式

def next_batch():
    img=np.array(sorted(glob.glob(r'./Datasets/Potsdam/train_images/*.png'))) # 影像保存路径，因为没有读取
    label=np.array(sorted(glob.glob(r'./Datasets/Potsdam/train_newlabels/*.png'))) # 语义分割类别标签保存路径
    #img=sorted(glob.glob(r'F:Users/wcf/English/Data/Vaihingen/train_images/*.png')) # 影像保存路径，因为没有读取
    #label=sorted(glob.glob(r'F:Users/wcf/English/Data/Vaihingen/train_labels/*.png'))
    #img = np.array('F:Users/wcf/English/data/Vaihingen/train_images/.png'))  # 影像保存路径
    #print(glob.glob(r'F:Users/wcf/English/data/Vaihingen/train_images/*.png'))
    #print(img)
    index=np.random.choice(len(img), batch_size)#??随机取样不可取，太多随机了
    x_batch=img[index]
    y_batch=label[index]
    image_batch,lab_batch=load_batch(x_batch, y_batch)
    return image_batch,lab_batch

def next_val_batch():
    img=np.array(sorted(glob.glob(r'E:/val_images/*.png'))) # 验证集影像保存路径
    label=np.array(sorted(glob.glob(r'E:/val_labels/*.png'))) # 验证集语义分割类别标签保存路径

    index=np.random.choice(len(img),batch_size)
    x_batch=img[index]
    y_batch=label[index]
    image_batch, lab_batch=load_batch(x_batch, y_batch)
    return image_batch, lab_batch