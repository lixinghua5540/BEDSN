# -*- coding: utf-8 -*-
"""
用于语义分割任务中：将RGB彩色类别图转换为类序号。 程序还有很大改进空间~
"""
import numpy as np
import  glob
import os
from skimage import io
import warnings
import sys
sys.path.append("./Semantic_Segmentation/Common_Codes/Train_Test")
warnings.filterwarnings("ignore")
from Parameters_Set import  categories,class_color

color_label_paths=np.array(sorted(glob.glob(r'./Semantic_Segmentation/Results/LANet/Vaihingen/RGB/*.png'))) # 类别标签路径#为什么之前写是png
print(color_label_paths)
file_num=1
for file_path in color_label_paths:
    name=os.path.basename(file_path)#修改一下格式，png转tif
    print(name)
    print(name.split('.'))
    print(f"Image{file_num}/{len(color_label_paths)}:{name} is running ^*^ ^o^ ^o^")
    color_label=io.imread(file_path)
    H,W,C=color_label.shape
    gray_label=np.zeros([H,W],dtype=np.uint8)
    for color_RGB,class_num in zip(class_color.values(),categories.keys()):
        gray_label[np.min((color_label==color_RGB),axis=-1)]=class_num  #只有每像素位置的RGB值均相等时才相等，即全部为True，最小为True
    io.imsave("./Semantic_Segmentation/Results/LANet/Vaihingen/Gray/"+name.split('.')[0]+".tif",gray_label)  # 类序号标签保存路径设置
    file_num+=1