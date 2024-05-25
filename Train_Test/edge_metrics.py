# -*- coding: utf-8 -*-
"""使用边界标签或膨胀的边界标签计算预测结果边界标签的准确程度"""
import  time
start_time=time.time()
import  numpy as np
import  glob
import pandas as pd
from collections import Counter
from skimage import  io
import  os
import tensorflow as tf

from datetime import timedelta
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


from Parameters_Set import *

#edge_assessment=
def cal_edge_metrics(save_root,edgeresults_path,edgelabel_path):
    session = tf.Session()
    TP=[]
    FN=[]
    FP=[]
    TN=[]
    TPD=[]
    FND=[]
    FPD=[]
    TND=[]
    for i in range(len(edgeresults_path)):#遍历每个影像
        save_path_assessment =save_root+"Plurality_Voting/edge_assessment/"#有问题
        if not os.path.exists(save_path_assessment):
            os.makedirs(save_path_assessment)
        results = io.imread(edgeresults_path[i])
        print(edgeresults_path[i])
        print(edgelabel_path[i])
        label = io.imread(edgelabel_path[i])
        label_tf=tf.convert_to_tensor(label)
        label_tf=tf.expand_dims(tf.expand_dims(label_tf,0),-1)
        D_label=tf.layers.max_pooling2d(label_tf,pool_size=[3,3],strides=[1,1],padding='SAME',data_format='channels_last')#看看是否正确
        D_label=np.squeeze(session.run(D_label))

        name_ = os.path.basename(edgeresults_path[i]).split(".")[0]
        #confu_list = cal_confu_matrix(label, results, numclasses)
        if i > 0:  print('\n')
        print(name_)
        print(results.shape)
        print(label.shape)
        print(results.dtype)
        print(label.dtype)
        print(results.max())
        print(label.max())
        print(results)
        print(label)
        TPvalue=np.zeros_like(label)
        FPvalue=np.zeros_like(label)
        FNvalue=np.zeros_like(label)
        TNvalue=np.zeros_like(label)
        TPDvalue=np.zeros_like(label)
        FPDvalue=np.zeros_like(label)
        FNDvalue=np.zeros_like(label)
        TNDvalue=np.zeros_like(label)
        #calculate TP
        inds_TP=(results==1) * (label==1)#logic
        TPvalue[inds_TP]=1
        TP_num=TPvalue.sum()
        print("inds_TP",inds_TP)
        print("TP_num",TP_num)
        #calculate FP
        inds_FP=(results==1) * (label==0)#unexpected result
        FPvalue[inds_FP]=1
        FP_num=FPvalue.sum()
        print("inds_FP",inds_FP)
        print("FP_num",FP_num)
        #calculate FN
        inds_FN=(results==0) * (label==1)#missing result
        FNvalue[inds_FN]=1
        FN_num=FNvalue.sum()
        print("inds_FN",inds_FN)
        print("FN_num",FN_num)
        #calculate TN
        inds_TN=(results==0) * (label==0)#只要前面为假就不在判断后面
        #inds_TN1=(results==0)
        #inds_TN2=(label==0)
        #TNvalue1[inds_TN1]=1#为什么全是True
        #TNvalue2[inds_TN2]=1
        TNvalue[inds_TN]=1
        TN_num=TNvalue.sum()
        print("inds_TN",inds_TN)
        print("TN_num",TN_num)
        TP.append(TP_num)
        FP.append(FP_num)
        FN.append(FN_num)
        TN.append(TN_num)
        recall=TP_num/(TP_num+FN_num)
        precision=TP_num/(TP_num+FP_num)
        #calculate TP of D
        indsD_TP=(results==1) * (D_label==1)#logic
        TPDvalue[indsD_TP]=1
        TPD_num=TPDvalue.sum()
        print("indsD_TP",indsD_TP)
        print("TPD_num",TPD_num)
        #calculate FP of D
        indsD_FP=(results==1) * (D_label==0)#unexpected result
        FPDvalue[indsD_FP]=1
        FPD_num=FPDvalue.sum()
        print("indsD_FP",indsD_FP)
        print("FPD_num",FPD_num)
        #calculate FN of D
        indsD_FN=(results==0) * (D_label==1)#missing result
        FNDvalue[indsD_FN]=1
        FND_num=FNDvalue.sum()
        print("indsD_FN",indsD_FN)
        print("FND_num",FND_num)
        #calculate TN of D
        indsD_TN=(results==0) * (D_label==0)
        TNDvalue[indsD_TN]=1
        TND_num=TNDvalue.sum()
        print("indsD_TN",indsD_TN)
        print("TND_num",TND_num)
        TPD.append(TPD_num)
        FPD.append(FPD_num)
        FND.append(FND_num)
        TND.append(TND_num)
        recall1=TPD_num/(TPD_num+FND_num)
        precision1=TPD_num/(TPD_num+FPD_num)
        #excel_confu_matrix(confu_list, save_path_assessment, name_)  # 生成混淆矩阵excel表格
        #metrics(confu_list, save_path_assessment, name_)  # 计算评估参数
        #accumu_confu_matrix += confu_list
    #计算总的

        with open(save_path_assessment + 'Edge_acc_all'  + '.txt', 'a', encoding='utf-8') as f:
            f.write('Image:\t%s\n' % name_)
            f.write('TP:\t%d\n' % TP_num)
            f.write('FP:\t%d\n' % FP_num)
            f.write('FN:\t%d\n' % FN_num)
            f.write('TN:\t%d\n' % TN_num)
            f.write('recall:\t%.4f\n' % (recall * 100))
            f.write('precision:\t%.4f\n' % (precision * 100))
            f.write('TPD:\t%d\n' % TPD_num)
            f.write('FPD:\t%d\n' % FPD_num)
            f.write('FND:\t%d\n' % FND_num)
            f.write('TND:\t%d\n' % TND_num)
            f.write('recall1:\t%.4f\n' % (recall1 * 100))
            f.write('precision1:\t%.4f\n' % (precision1 * 100))
    with open(save_path_assessment + 'Edge_acc_all'  + '.txt', 'a', encoding='utf-8') as f:
        f.write('All metrics:\t\n' )
        f.write('TPall:\t%d\n' % (sum(TP)))
        f.write('FPall:\t%d\n' % (sum(FP)))
        f.write('FNall:\t%d\n' % (sum(FN)))
        f.write('TNall:\t%d\n' % (sum(TN)))
        recall_all=sum(TP)/(sum(TP)+sum(FN))
        precision_all=sum(TP)/(sum(TP)+sum(FP))
        f.write('recall:\t%.4f\n' % (recall_all * 100))
        f.write('precision:\t%.4f\n' % (precision_all * 100))
        f.write('TPallD:\t%d\n' % (sum(TPD)))
        f.write('FPallD:\t%d\n' % (sum(FPD)))
        f.write('FNallD:\t%d\n' % (sum(FND)))
        f.write('TNallD:\t%d\n' % (sum(TND)))
        recall_allD=sum(TPD)/(sum(TPD)+sum(FND))
        precision_allD=sum(TPD)/(sum(TPD)+sum(FPD))
        f.write('recall1:\t%.4f\n' % (recall_allD * 100))
        f.write('precision1:\t%.4f\n' % (precision_allD * 100))

if __name__ == "__main__":
    #network='BEDSN'
    dataset='Potsdam'
    datasource='./Semantic_Segmentation/Results/'+dataset+'/'
    net_list=os.listdir(datasource)
    print(net_list)
    for i in range(len(net_list)):
        if i!=1:
            save_root= datasource+net_list[i]+'/'
            path_GT_edge_labels='./Datasets/'+dataset+'/test_newedges/'
            edgeresults_path=np.array(sorted(glob.glob(save_root+"Plurality_Voting/Edge/*.tif"))) # 预测结果
            edgelabel_path=np.array(sorted(glob.glob(path_GT_edge_labels+"*.tif")))  # 边界腐蚀真值标签
            cal_edge_metrics(save_root,edgeresults_path,edgelabel_path)
    #if len(results_path) > 1:#对于边界的统计采用什么样的策略，也是每一个的像元级别数量，而不是算数平均
    #    print('\n')
    #    excel_confu_matrix(accumu_confu_matrix, save_path_assessment)
    #    metrics(accumu_confu_matrix, save_path_assessment, 'all')
    #print(get_time_dif(start_time))