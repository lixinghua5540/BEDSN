# -*- coding: utf-8 -*-
"""几种常见的数据增强方法"""

import numpy as np
import random
from PIL import  Image
import matplotlib.pyplot as plt

""" 图片的旋转、裁剪与翻转 """

foldername_s="F:/Users/wcf/English/Data/Vaihingen/"#F:\Users\wcf\2022-王彩凤-毕业提交材料\6_王彩凤-已收集程序&工具&数据\3_数据\2-ISPRS 2D Potsdam数据集\2_Ortho_RGB\train

# dataset=['2_10','2_11','2_12','3_10','3_11','3_12',
#                     '4_10','4_11','4_12','5_10','5_11','5_12',
#                     '6_7','6_8','6_9','6_10','6_11','6_12',
#                     '7_7','7_8','7_9','7_10','7_11','7_12']
#
#dataset=[1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]
dataset=[2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]
# dataset=['2_13','2_14','3_13','3_14',
#                      '4_13','4_14','4_15','5_13','5_14','5_15',
#                      '6_13','6_14','6_15',
#                      '7_13']
for picture in dataset:
    picture=str(picture)
    image_train=Image.open("F:/Users/wcf/English/Data/Vaihingen/top/test/top_mosaic_09cm_area"+picture+".tif")
    label_train=Image.open("F:/Users/wcf/English/Data/Vaihingen/GT/top_mosaic_09cm_area"+picture+".tif")
    newlabel_train=Image.open("F:/Users/wcf/English/Data/Vaihingen/GT_onehot/top_mosaic_09cm_area"+picture+".tif")

    print(f"这是图:{picture}")
    image_width = image_train.size[0]
    image_height= image_train.size[1]
    #crop_win_size =512#512的patch？
    crop_win_size = 256  # 512的patch？

    for i in range(1,201):#

        x0=random.randint(0,image_width-crop_win_size)#这个取样策略非常炸裂
        y0=random.randint(0,image_height-crop_win_size)
        x1=x0+crop_win_size   # x1=x0+crop_win_size  引入随机尺度变化的尺度
        y1=y0+crop_win_size
        box=(x0,y0,x1,y1)
        imta=image_train.crop(box) # 影像
        laba=label_train.crop(box) # RGB影像
        newlaba=newlabel_train.crop(box) # label影像
        imta.save(foldername_s+"test_images/vaihingen_"+picture+"_"+str(i)+".png")
        laba.save(foldername_s+"test_labels/vaihingen_"+picture+"_"+str(i)+".png")
        newlaba.save(foldername_s+"test_newlabel2/vaihingen_"+picture+"_"+str(i)+".png")

        imgout=imta.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        labout=laba.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        newlabout=newlaba.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转

        imgout.save(foldername_s+"test_images/vaihingen_"+picture+"_"+str(i)+"_1.png")
        labout.save(foldername_s+"test_labels/vaihingen_"+picture+"_" + str(i)+"_1.png")
        newlabout.save(foldername_s+"test_newlabel2/vaihingen_"+picture+"_" + str(i)+"_1.png")

        for j in range(1,4):
            random_angle=j*90
            imb=imta.rotate(random_angle)
            lab=laba.rotate(random_angle)
            newlab=newlaba.rotate(random_angle)

            imb.save(foldername_s+"test_images/vaihingen_"+picture+"_"+str(i)+"_"+str(j+1)+".png")
            lab.save(foldername_s+"test_labels/vaihingen_"+picture+"_"+str(i)+"_"+str(j+1)+".png")
            newlab.save(foldername_s+"test_newlabel2/vaihingen_"+picture+"_"+str(i)+"_"+str(j+1)+".png")

