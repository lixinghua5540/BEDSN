# -*- coding: utf-8 -*-

batch_size=16#default16
if_training=False

if if_training==False:
    keep_prob_dropopt=1.0
    batch_size=1
else:
    keep_prob_dropopt=0.8
    if batch_size==1:
        print("Warning:请修改batch_size的大小！！！！！！")

"""
训练涉及参数
"""
train_epoches=60
crop_image_size=1024 #1024
crop_overlap_size=512 # 512
out_stride=32  # 网络采样的倍数 32

numclasses=6
dataset="Vaihingen"
categories={0:"Clutter/Background",1:"Building",2:"Imp_Surf",3:"Low_veg",4:"Tree",5:"Car"}  # 标签-对应类别映射关系
class_color={"Clutter/Background":[255,0,0],"Building":[0,0,255],"Imp_Surf":[255,255,255],"Low_veg":[0,255,255],"Tree":[0,255,0],"Car":[255,255,0]}  # 设置标签颜色（这里是6种）
root_name="Models"  # 根据需要调整#原为semantic Segmentation_models

path_testimgs=r"./Datasets/Potsdam/test_images/"
path_GTlabels=r"./Datasets/Potsdam/test_newlabels/"   # 测试集真值标签路径
path_GT_noboundary_labels=r"./Datasets/Potsdam/test_newlabel_noboundary/"   # 测试集边界腐蚀标签路径(用于ISPRS 数据集精度计算，针对其他数据集需适当调整)
#path_GT_edge=r"./Semantic_Segmentation/Dataset/test_edge_labels/"#先不用

Total_Images=16000#这 有什么用，是读不出来还是怎么的？
num_batches=Total_Images//batch_size  # total number of train imagenet-simple-labels.json

Networks_dic={ 'UNet':['UNet','UNet'],            # 模型网络名称 ：【"文件夹名",'模型操作名称'】
                                'SegNet': ['SegNet', 'SegNet'],
                                'SCAttNet': ['SCAttNet', 'SCAttNet_SegNet'],
                                'DeepLabv3_plus': ['DeepLabv3_plus', 'DeepLabv3_plus'],
                                'ResUNet_a': ['ResUNet_a', 'ResUNet_a_d6'],
                                'FCN_8s': ['FCN_8s', 'FCN_8s'],
                                'ERN': ['ERNet', 'ERNet'],
                                'GAMNet':['GAMNet','GAMNet_resnet101'],
                                'BEDSN': ['BEDSN', 'BEDSN'],
                                'Lighter_BEDSN': ['LBEDSN', 'LBEDSN'],
                                'Parallel_BEDSN': ['PBEDSN', 'PBEDSN'],

              }

"""测试参数"""
record_epoch=60 #"''Results_60epoch"存放结果为256x256推理结果
Name_select_network='SegNet'
save_root='./Semantic_Segmentation/Results/'+dataset+'/'+Name_select_network+'/' + str(16)+'/'             # 模型所在根目录，根据需要调整
#save_root='F:/Users/wcf/English/Semantic_Segmentation/Models/'+Name_select_network+'/'                    # 模型所在根目录，根据需要调整
save_path_predict=save_root+"Results_"+str(record_epoch)+"epoch/Semantic_Segmentation/"  # 预测类别保存路径
save_path_predictRGB=save_root+"Results_"+str(record_epoch)+"epoch/RGB/"                 # 预测类别标签ＲＧＢ保存路径
save_path_assement=save_root+"Results_"+str(record_epoch)+"epoch/Evaluation_metrics/"    # 精度评定结果保存路径
