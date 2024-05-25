# -*- coding: utf-8 -*-
# # 数字标签转彩色标签
import numpy as np
import glob
import os
import imageio
from skimage import  io
import sys
sys.path.append("./Semantic_Segmentation/Common_Codes/Train_Test")
from Parameters_Set import  save_path_predict,save_path_predictRGB,class_color

# 数字标签路径
if __name__=="__main__":
    #path = np.array(sorted(glob.glob(save_path_predict + "*.tif")))
    path = np.array(sorted(glob.glob("./Semantic_Segmentation/Results/MDANet/VH/Plurality_Voting/Semantic_Segmentation/" + "*.tif")))
    save_path_predictRGB='./Semantic_Segmentation/Results/MDANet/VH/Plurality_Voting/RGB/'#新添加的
    if not os.path.exists(save_path_predictRGB): os.mkdir(save_path_predictRGB)
    for file in range(len(path)):
        name = os.path.basename(path[file])
        print(f"Image{file + 1}/{len(path)}:{name} is running......")
        label = io.imread(path[file])
        h, w = label.shape
        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for i, rgb in zip(range(len(class_color)), class_color.values()):
            # print(i,rgb) # 数字对应颜色
            label_rgb[label == i] = rgb
        # 保存图片
        imageio.imsave(save_path_predictRGB + name, label_rgb)