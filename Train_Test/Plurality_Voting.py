# -*- coding: utf-8 -*-
"""
基于集成学习思想，该程序将多个epoch模型下的语义分割结果进行多数投票，获取最终语义分割结果
"""
from collections import Counter
import numpy as np
import os
import glob
import imageio
from skimage import  io
from Parameters_Set import save_path_predict,save_root,class_color

from Evaluate_operation import *
operations=['Vote','Evaluation','RGB_transform']
operation_select=input("operations_select:     1: Vote or    2: Evaluation or    3:RGB_transform-->  ")

if operation_select=="1":
    data_last_dir = "Semantic_Segmentation"
    data_name_files = np.array(sorted(glob.glob(save_path_predict + '*.tif')))  # 用于获取测试影像的名称

    for _ in range(len(data_name_files)):
        Multi_Seg = []  # 多个epoch模型参数的分类结果存储器 在每个像素像进行拼接
        data_name = os.path.basename(data_name_files[_])
        print(f"{_ + 1}: Image of  {data_name} is Running!")
        assist_countNumber = 0  # 辅助计数：当为1时标签只有一个epoch投票，>1时Multi_Seg将所有投票结果进行拼接
        for i in range(56,61):  # 实验中根据后5个训练轮次保存的语义分割模型进行集成。如采用多epoch模型集成思想，可加以调整
            assist_countNumber+= 1
            dat_mid_dir = "Results_" + str(i) + "epoch"
            path_name = os.path.join(save_root, dat_mid_dir, data_last_dir, data_name)
            seg_ = io.imread(path_name)
            multi_seg = np.expand_dims(seg_, axis=-1)
            if assist_countNumber == 1:
                Multi_Seg = multi_seg
            else:
                Multi_Seg = np.concatenate([Multi_Seg, multi_seg], axis=-1)  # 来自多张影像堆叠

        Result = np.zeros(Multi_Seg.shape[:2], dtype=np.int8)

        for i, i_ in enumerate(Multi_Seg):
            for j, j_ in enumerate(i_):
                static_result = Counter(j_)  # 针对多个epoch对每个像素的分类结果进行统计
                Result[i][j] = max(static_result, key=static_result.get)
        #imageio.imsave(save_root + "Results/Plurality_Voting/Semantic_Segmentation/" + data_name, Result)
        Save_path = save_root + "Plurality_Voting/Semantic_Segmentation/"
        if not os.path.exists(Save_path):
            os.makedirs(Save_path)
        imageio.imsave(Save_path + data_name, Result)
elif operation_select=='2':   # 利用多epochs的集成结果 进行语义分割精度评估
    results_path_=np.array(sorted(glob.glob(save_root+"Results/Plurality_Voting/Semantic_Segmentation/*.tif")))
    evaluation_path_=save_root+"Results/Plurality_Voting/Evaluation_metrics/"
    for i in range(len(results_path_)):
        confu_list = []
        results=io.imread(results_path_[i])
        label=io.imread(oris_path[i])
        confu_list = cal_confu_matrix(label, results, numclasses)
        name_=os.path.basename(results_path_[i]).split(".")[0]
        if i > 0:  print('\n')
        print(name_)
        excel_confu_matrix(confu_list, evaluation_path_, name_)  # 生成混淆矩阵excel表格
        metrics(confu_list, evaluation_path_, name_)  # 计算评估参数

        accumu_confu_matrix += confu_list

    if len(results_path_)>1:
        print('\n')
        excel_confu_matrix(accumu_confu_matrix, evaluation_path_)
        metrics(accumu_confu_matrix, evaluation_path_, 'all')
    print(get_time_dif(start_time))
else:  # 进行语义分割结果的RGB着色
    results_path_=np.array(sorted(glob.glob(save_root+"Results/Plurality_Voting/Semantic_Segmentation/*.tif")))
    RGB_results_path=save_root+"Results/Plurality_Voting/RGB/"
    for file in range(len(results_path_)):
        name = os.path.basename(results_path_[file])
        print(f"Image{file+1}/{len(results_path_)}:{name} is running......")
        label=io.imread(results_path_[file])
        h, w = label.shape
        label_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i, rgb in zip(range(len(class_color)), class_color.values()):
            label_rgb[label == i] = rgb
        imageio.imsave(RGB_results_path+name, label_rgb)
