# -*- coding: utf-8 -*-
"""使用腐蚀标签计算语义分割精度"""
import  time
start_time=time.time()
import  numpy as np
import  glob
import pandas as pd
from collections import Counter
from skimage import  io
import  os

from datetime import timedelta
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

from Parameters_Set import *
network='MDANet'
dataset='PD'
save_root='./Semantic_Segmentation/Results/'+network+'/'+dataset+'/'
results_path=np.array(sorted(glob.glob(save_root+"Plurality_Voting/Semantic_Segmentation/*.tif"))) # 预测结果
oris_path=np.array(sorted(glob.glob(path_GT_noboundary_labels+"*.tif")))  # 边界腐蚀真值标签


def cal_confu_matrix(label, predict, class_num):  # 可由tensorflow自带函数进行计算
    predict = predict[np.where(label < class_num)]  # 用于祛除腐蚀部分像素
    label = label[np.where(label < class_num)]

    confu_list = []
    for i in range(class_num):
        c = Counter(label[np.where(predict == i)])  # 获取每一次实际实际类别对应的位置，并对对应位置域内的预测值进行计数                                                   # 如A = Counter({1：2，4：2})则访问A时，只有A[1], A[4]有值，其余均为0
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int64)

def excel_confu_matrix(confu_list, save_path='./', save_name=None):
    """
    用于将混淆矩阵保存为excel表格
    :param confu_list:  混淆矩阵
    :param save_path: 保存路径
    :param save_name:
    :return:
    """
    data_df = pd.DataFrame(confu_list)
    data_df.columns = list(categories.values())
    data_df.index = list(categories.values())
    if save_name!=None:    # 此处用于判断是单个影像的混淆矩阵，还是整个测试集的混淆矩阵。可根据需要加以删减修改~
        writer = pd.ExcelWriter(save_path + "测试集混淆矩阵_" + save_name + ".xlsx")
    else:
        writer = pd.ExcelWriter(save_path + "测试集混淆矩阵_整体.xlsx")
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()


def metrics(confu_mat_total, save_path='./', save_name=None):
    """
      用于 语义分割精度计算
      :param confu_mat_total: 总混淆矩阵
      :param save_path: 保存地址
      :param save_name: 保存名称
      :return:  txt输出混淆矩阵, precision，recall，IOU，f-score
    """
    class_num = confu_mat_total.shape[0]
    confu_mat = confu_mat_total.astype(np.float64) + 1e-15
    col_sum = np.sum(confu_mat, axis=1)  # 按行求和
    raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

    '''计算各类面积比，以求OA值'''
    OA = 0
    for i in range(class_num):
        OA = OA + confu_mat[i, i]  # 对角线上的值
    OA = OA / confu_mat.sum()
    print('OA: ', OA)

    '''Kappa'''
    pe_fz = 0
    for i in range(class_num):
        pe_fz += col_sum[i] * raw_sum[i]
    pe = pe_fz / (np.sum(confu_mat) * np.sum(confu_mat))
    Kappa = (OA - pe) / (1 - pe)  # 根据公式 化
    print('Kappa: ', Kappa)

    TP = []  # 识别中每类分类正确的个数
    for i in range(class_num):
        TP.append(confu_mat[i, i])

    TP = np.array(TP)
    FN = raw_sum - TP
    FP = col_sum - TP

    # 计算并写出precision，recall, F1_Score，AF以及MIOU
    F1_m = []
    IoU_m = []

    for i in range(class_num):
        F1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i]) # 计算每类的F1_Score
        F1_m.append(F1)
        IoU = TP[i] / (TP[i] + FP[i] + FN[i])
        IoU_m.append(IoU)

    F1_m = np.array(F1_m)
    IoU_m = np.array(IoU_m)
    N = raw_sum / np.sum(raw_sum)
    percents = np.array(N)  # 每类占比

     #"""不算背景精度OA"""
    OA_ = 0
    for i in range(1, class_num):
        OA_ = OA_ + confu_mat[i, i]  # 对角线上的值
    OA_ = (OA_ / confu_mat.sum()) / (1 - percents[0])
    print('不算背景精度OA: ', OA_)

    if save_name is not None:
        with open(save_path + 'Accuracy_' + save_name + '.txt', 'a', encoding='utf-8') as f:

            f.write('OA:\t%.4f\n' % (OA * 100))
            f.write('Kappa:\t%.4f\n' % (Kappa * 100))
            f.write('AF:\t%.4f\n' % (np.mean(F1_m[1:]) * 100))
            f.write('MIoU:\t%.4f\n' % (np.mean(IoU_m[1:]) * 100))
            f.write('noboundary_OA:\t%.4f\n' % (OA_ * 100))
            # 影像中每类的占比
            f.write('\n')
            f.write('-------------------------------------------------\n')
            for i in range(class_num):
                f.write(categories[i] + ':%.4f\n' % (percents[i] * 100))
            f.write('\n')
            f.write('-------------------------------------------------\n')

            # 各类精确率指标
            f.write('Precision:\n')
            for i in range(class_num):
                f.write(categories[i] + ':%.4f\t' % (float(TP[i] / col_sum[i]) * 100))
            f.write('\n')

            # 各类召回率指标
            f.write('Recall:\n')
            for i in range(class_num):
                f.write(categories[i] + ':%.4f\t' % (float(TP[i] / raw_sum[i]) * 100))
            f.write('\n')

            # 各类F1得分指标
            f.write('F1_score:\n')
            for i in range(class_num):
                f.write(categories[i] + ':%.4f\t' % (float(F1_m[i]) * 100))
            f.write('\n')

            # 各类交并比指标
            f.write('IoU:\n')
            for i in range(class_num):
                f.write(categories[i] + ':%.4f\t' % (float(IoU_m[i]) * 100))
            f.write('\n\n')
        f.close()

    # AF = (np.mean(F1_m[1:])) * 100
    # MIoU = (np.mean(IoU_m[1:])) * 100
    # print('AF: ', AF)
    # print("MIoU: ", MIoU)


if __name__ == "__main__":
    accumu_confu_matrix = np.zeros([numclasses, numclasses], dtype=np.int64)
    for i in range(len(results_path)):
        save_path_assessment =save_root+"Noboundary_assessment/"
        if not os.path.exists(save_path_assessment):
            os.makedirs(save_path_assessment)
        results = io.imread(results_path[i])
        print(results_path[i])
        label = io.imread(oris_path[i])
        print(oris_path[i])
        name_ = os.path.basename(results_path[i]).split(".")[0]
        confu_list = cal_confu_matrix(label, results, numclasses)
        if i > 0:  print('\n')
        print(name_)
        excel_confu_matrix(confu_list, save_path_assessment, name_)  # 生成混淆矩阵excel表格
        metrics(confu_list, save_path_assessment, name_)  # 计算评估参数
        accumu_confu_matrix += confu_list

    if len(results_path) > 1:
        print('\n')
        excel_confu_matrix(accumu_confu_matrix, save_path_assessment)
        metrics(accumu_confu_matrix, save_path_assessment, 'all')
    print(get_time_dif(start_time))
