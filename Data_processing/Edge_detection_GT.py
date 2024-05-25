# -*- coding: utf-8 -*-
"""利用语义分割真值标签获取边缘检测参考标签
    基本思想：四邻域辅助判断   """

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
#from Parameters_Set import  path_GTlabels,path_GT_edge
def cal_edge(pathname):
    Images=np.array(sorted(glob.glob(pathname["label_path"]+"/*.tif")))

    num=0
    for filename in Images:
        num+=1
        print(f"{num}: {filename}")
        img=io.imread(filename)
        img_size=np.array(img.shape)
        High=img_size[0]
        Width = img_size[1]
    
        edge_Value=np.zeros([High, Width],np.uint8)
        imgbigger=np.zeros([High+2,Width + 2],np.uint8)
    
        name=os.path.basename(filename)
        # --------------扩张矩阵
        imgbigger[0,1:Width+1]=img[0,:]
        imgbigger[High+1,1:Width+1]=img[-1,:]
        imgbigger[1:High+1,0]=img[:,0]
        imgbigger[1:High+1,Width+1]=img[:,-1]
        imgbigger[1:-1,1:-1]=img
    
        for i in range(1, High + 1):
            print(f"{num}<---{i}............")
            for j in range(1, Width + 1):
                if max([imgbigger[i][j], imgbigger[i][j-1], imgbigger[i][j+1], imgbigger[i-1][j],imgbigger[i+1][j]]) !=\
                   min([imgbigger[i][j], imgbigger[i][j-1], imgbigger[i][j+1], imgbigger[i-1][j],imgbigger[i+1][j]]):
                    edge_Value[i - 1][j - 1] = 1
    
        imagest=Image.fromarray(edge_Value)
        imagest.save(pathname["edge_path"]+"/"+name)
datasource='./Datasets/Potsdam/'
file_list=os.listdir(datasource)
print(file_list)
#for i in range(len(file_list)):
    #if i!=1:
#print("network:",file_list[i])
#path_GTlabels=datasource+file_list[i]+'/Plurality_Voting/Semantic_Segmentation'
path_GTlabels=datasource+'test_newlabels'
print(path_GTlabels)
#path_GT_edge=datasource+file_list[i]+'/Plurality_Voting/Edge'
path_GT_edge=datasource+'test_newedges'
if not os.path.exists(path_GT_edge): os.mkdir(path_GT_edge)
pathname={"label_path":path_GTlabels,"edge_path":path_GT_edge}
cal_edge(pathname)

