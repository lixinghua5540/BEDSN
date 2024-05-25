# -*- coding: utf-8 -*-

import  time
start_time=time.time()
import  numpy as np
import  glob
import pandas as pd
from collections import Counter
from skimage import  io
import  os
from cau_time import get_time_dif
from Parameters_Set import save_root,path_GTlabels,numclasses,categories
import tensorflow as tf

#save_path_predict=save_root+"Plurality_Voting/Semantic_Segmentation/"
save_path_predict="./Semantic_Segmentation/Results/LANet/Vaihingen/Gray/"
#save_path_assement=save_root+"Plurality_Voting/Evaluation_metrics/"    # 精度评定结果保存路径
save_path_assement="./Semantic_Segmentation/Results/LANet/Vaihingen/Evaluation_metrics/" 
##没有则创建文件夹
results_path=np.array(sorted(glob.glob(save_path_predict+"*.tif")))  # 预测结果
oris_path=np.array(sorted(glob.glob(path_GTlabels+"*.tif")))               # 真值标签

accumu_confu_matrix=np.zeros([numclasses,numclasses],dtype=np.int64)

def cal_confu_matrix(label,predict,class_num):  #可由tensorflow自带函数进行计算
    """
    根据预测结果与真值标签获取混淆矩阵
    :param label:  真值标签
    :param predict: 预测结果
    :param class_num:  语义分割类别数
    :return:  混淆矩阵
    """
    confu_list=[]
    for i in range(class_num):#去除clutter类别
        c=Counter(label[np.where(predict==i)])    # 获取每一次实际实际类别对应的位置，并对对应位置域内的预测值进行计数
        single_row=[]
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)

def excel_confu_matrix(confu_list,save_path='./',save_name=None):
    """
    用于将混淆矩阵保存为excel表格
    :param confu_list:  混淆矩阵
    :param save_path: 保存路径
    :param save_name: 保存名字
    :return:
    """
    data_df=pd.DataFrame(confu_list)
    data_df.columns=list(categories.values())
    data_df.index=list(categories.values())
    if save_name!=None:    # 此处用于判断是单个影像的混淆矩阵，还是整个测试集的混淆矩阵。可根据需要加以删减修改~
        writer=pd.ExcelWriter(save_path+"测试集混淆矩阵_"+save_name+".xlsx")
    else:
        writer=pd.ExcelWriter(save_path+"测试集混淆矩阵_整体.xlsx")
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()

def metrics(confu_mat_total, save_path='./',save_name=None):
    """
    用于 语义分割精度计算
    :param confu_mat_total: 总混淆矩阵
    :param save_path: 保存地址
    :param save_name: 保存名称
    :return:  txt输出混淆矩阵, precision，recall，IOU，f-score
    """
    class_num=confu_mat_total.shape[0]
    print("C",class_num)
    confu_mat=confu_mat_total.astype(np.float32)+1e-15#去除了0值干啥
    col_sum=np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum=np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    OA =0
    for i in range(class_num):#去除clutter类别
        OA=OA+confu_mat[i, i]
    OA=OA/confu_mat.sum()  # 总体语义分割精度
    print('OA: ',OA)

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz+=col_sum[i] * raw_sum[i]
    pe=pe_fz/(np.sum(confu_mat)*np.sum(confu_mat))
    Kappa = (OA-pe) / (1+1e-15- pe)    # Kappa系数计算
    print('Kappa: ', Kappa)

    TP=[]
    for i in range(class_num):
        TP.append(confu_mat[i, i])

    TP=np.array(TP)
    FN=raw_sum-TP
    FP=col_sum-TP

    # 计算并写出precision，recall, F1_Score，AF以及MIOU
    F1_m=[]
    IoU_m=[]

    for i in range(class_num):
        F1=TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i]) # 计算每类的F1_Score
        F1_m.append(F1)
        IoU=TP[i] /(TP[i] + FP[i] + FN[i])
        IoU_m.append(IoU)

    F1_m = np.array(F1_m)
    IoU_m = np.array(IoU_m)
    N=raw_sum/np.sum(raw_sum)
    percents=np.array(N)  # 每类占比

    # """不算背景精度OA"""
    OA_ = 0
    for i in range(1, class_num):
         OA_ = OA_+confu_mat[i, i]  # 对角线上的值
    OA_=(OA_/confu_mat.sum())/(1-percents[0])
    print(confu_mat.sum())
    print(raw_sum)
    print(percents[0])
    print('不算背景精度OA: ', OA_)

    if save_name is not None:
        with open(save_path+'Accuracy_'+save_name+'.txt', 'w',encoding='utf-8') as f:

            f.write('OA:\t%.4f\n' % (OA*100))
            f.write('AF:\t%.4f\n'%(np.mean(F1_m[1:])*100))  # AF算时不将背景部分计算入内
            f.write('Kappa:\t%.4f\n'%(Kappa*100))
            f.write('MIoU:\t%.4f\n'%(np.mean(IoU_m[1:])*100)) # MIoU计算时亦不将背景部分计算入内

            # 影像中每类的占比
            f.write('\n')
            f.write('-------------------------------------------------\n')
            for i in range(class_num):
                f.write(categories[i]+':%.4f\n' % (percents[i] * 100))
            f.write('\n')
            f.write('-------------------------------------------------\n')

            # 各类精确率指标
            f.write('Precision:\n')
            for i in range(class_num):
                f.write(categories[i]+':%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 各类召回率指标
            f.write('Recall:\n')
            for i in range(class_num):
                f.write(categories[i]+':%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 各类F1得分指标
            f.write('F1_score:\n')
            for i in range(class_num):
                f.write(categories[i]+':%.4f\t' % (float(F1_m[i]) * 100))
            f.write('\n')
            # 各类交并比指标
            f.write('IoU:\n')
            for i in range(class_num):
                f.write(categories[i]+':%.4f\t'%(float(IoU_m[i]) * 100))
            f.write('\n')
        f.close()

    # AF = (np.mean(F1_m[1:])) * 100
    # MIoU = (np.mean(IoU_m[1:])) * 100
    # print('AF: ', AF)
    # print("MIoU: ", MIoU)

if __name__=="__main__":
    accumu_confu_matrix = np.zeros([numclasses, numclasses], dtype=np.int64)
    for i in range(len(results_path)):
        results = io.imread(results_path[i])
        label = io.imread(oris_path[i])
        confu_list = cal_confu_matrix(label, results, numclasses)
        name_ = os.path.basename(results_path[i]).split(".")[0]
        if i > 0:  print('\n')
        print(name_)
        excel_confu_matrix(confu_list, save_path_assement, name_)  # 生成混淆矩阵excel表格
        metrics(confu_list, save_path_assement, name_)  # 计算评估参数
        accumu_confu_matrix += confu_list

    if len(results_path) > 1:  # 整个测试集的语义分割性能评估
        print('\n')
        excel_confu_matrix(accumu_confu_matrix, save_path_assement)
        metrics(accumu_confu_matrix, save_path_assement, 'all')

    print(get_time_dif(start_time))


