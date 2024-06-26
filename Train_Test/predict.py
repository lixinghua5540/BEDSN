# -*- coding: utf-8 -*-
"""采用重叠推理进行测试集大影像的预测"""
import numpy as np
import glob
from skimage import io
import math
import importlib
from cau_time import get_time_dif
import warnings
warnings.filterwarnings("ignore")
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] ="1"
import tensorflow as tf
print(tf.test.is_gpu_available())
from Parameters_Set import *
assert if_training==0,"if_training setting error!!!"
import time
now_time = time.time()
record_epoch=sys.argv[1]
record_epoch=int(record_epoch)

save_path_predict=save_root+"Results_"+str(record_epoch)+"epoch/Semantic_Segmentation/"  # 预测类别保存路径
save_path_predictRGB=save_root+"Results_"+str(record_epoch)+"epoch/RGB/"                 # 预测类别标签ＲＧＢ保存路径
save_path_assement=save_root+"Results_"+str(record_epoch)+"epoch/Evaluation_metrics/"    # 精度评定结果保存路径
dir_name=Networks_dic[Name_select_network][0]   # 文件夹名称
Name_operation=Networks_dic[Name_select_network][1]  # 模型操作名称
Name_network='.'.join([root_name,dir_name,Name_select_network]) # 模型文件绝对地址【从父目录开始】
module_name=importlib.import_module(Name_network)  # 获取模型文件
inference_model=getattr(module_name,Name_operation) # 导入模型中的操作函数
batch_size=1
if_training=False

def downsampe_intpicture(img, downsample_times):
    Img_shape = np.array(img.shape)
    height = Img_shape[0]
    width = Img_shape[1]

    Height_convert = math.ceil(height / downsample_times) * downsample_times  #
    Width_convert = math.ceil(width / downsample_times) * downsample_times
    Img_shape[0] = Height_convert
    Img_shape[1] = Width_convert
    Img = np.zeros(Img_shape, dtype=img.dtype)  # 转换后的影像矩阵
    diff_height = Height_convert -height
    diff_width = Width_convert-width

    if diff_height == 0 and diff_width == 0:
        Img=img
        box=(0, None, 0, None)
        return False,Img,box
    else:
        supple_upheight = int(diff_height / 2)
        supple_downheight = math.ceil(diff_height / 2)
        supple_leftwidth = int(diff_width / 2)
        supple_rightwidth = math.ceil(diff_width / 2)

        """     高度方向上的扩充        """
        Img[:supple_upheight, supple_leftwidth:width + supple_leftwidth, :]=np.tile(img[0, :, :], (supple_upheight, 1, 1))  # tile扩充函数，从而使得其一一对应
        Img[supple_upheight:supple_upheight + height, supple_leftwidth:width + supple_leftwidth, :] = img
        Img[supple_upheight+height:,supple_leftwidth:width + supple_leftwidth, :] = np.tile(img[-1, :, :], (supple_downheight, 1, 1))


        """   宽度方向上的扩充  """
        Img[:, :supple_leftwidth, :]=np.tile(np.expand_dims(Img[:,supple_leftwidth, :], axis=1),(1, supple_leftwidth, 1))
        Img[:, supple_leftwidth + width:,:]=np.tile(np.expand_dims(Img[:,supple_leftwidth+width-1,:], axis=1),(1, supple_rightwidth, 1))
        box=(supple_upheight, supple_downheight, supple_leftwidth, supple_rightwidth)  # 返回几个关键扩充点，以备后面进行裁剪需要
        return True,Img, box

def image_covery(img,box):
    supple_upheight, supple_downheight, supple_leftwidth, supple_rightwidth = box
    recovery_img=img[supple_upheight:img.shape[0]-supple_downheight,supple_leftwidth:img.shape[1]-supple_rightwidth]   # 恢复原始影像大小
    return  recovery_img

img = tf.placeholder(tf.float32, [batch_size, None,None,3])
test_img=sorted(glob.glob(path_testimgs+"*.tif"))
phase_train=tf.placeholder(tf.bool, name='phase_train')
pred,_=inference_model(img, phase_train) # 输出结果包含语义分割、边缘检测两项内容
# pred=inference_model(img, phase_train) # 输出仅包含语义分割结果
saver=tf.train.Saver()

def save():
    tf.global_variables_initializer().run()
    checkpoint_dir=save_root+'checkpoint/'
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
    print(ckpt)#None?
    ckpt_num=['74648','75981','77314','78647','79980']
    if ckpt and ckpt.model_checkpoint_path:
        #ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(record_epoch)
        print(num_batches/8)#为什么除以8
        #ckpt_name='BEDSN.ckpt-'+str(int(record_epoch*num_batches/2))
        ckpt_name='BEDSN.ckpt-'+ckpt_num[record_epoch-56]
        print(ckpt_name)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))#读文件

    for num_img in range(0, len(test_img)):#一整景
        file_name=test_img[num_img]
        print(file_name,"   ^o^ ^o^")
        name=os.path.basename(file_name) # 作用同name=file_name.split('/')[-1]
        x_batch=io.imread(file_name)/ 255.0
        number_row=math.ceil((x_batch.shape[1]-crop_image_size)/(crop_image_size-crop_overlap_size)+1)  #横向前进次数
        number_col=math.ceil((x_batch.shape[0]-crop_image_size)/(crop_image_size-crop_overlap_size)+1)   #纵向前进次数
        """选取影像块"""
        for i in range(number_col):
            Result_row=[] # 遍历横向结果
            if i==0:
                location_up=0
                result_up=0
                if i==number_col-1:  # 影像高不大于设定大小
                    location_down=x_batch.shape[0]
                    result_down=location_down-location_up+1
                else:
                    location_down=crop_image_size
                    result_down=location_down-int(crop_overlap_size * 0.5)
            else:
                location_up=i*(crop_image_size-crop_overlap_size)
                result_up=int(crop_overlap_size*0.5)
                if i==number_col-1:
                    location_down=x_batch.shape[0]
                    result_down=location_down-location_up+1
                else:
                    location_down=location_up+crop_image_size
                    result_down=crop_image_size-int(crop_overlap_size * 0.5)

            for j in range(number_row):
                if j==0:
                    location_left=0
                    result_left=0
                    if j == number_row - 1:
                        location_right=x_batch.shape[1]
                        result_right=location_right-location_left+1
                    else:
                        location_right=crop_image_size
                        result_right=crop_image_size-int(crop_overlap_size*0.5)

                else:
                    location_left=j*(crop_image_size-crop_overlap_size)
                    result_left=int(crop_overlap_size * 0.5)
                    if j==number_row-1:
                        location_right=x_batch.shape[1]
                        result_right=location_right-location_left+1
                    else:
                        location_right=location_left+crop_image_size
                        result_right=crop_image_size-int(crop_overlap_size * 0.5)
                """对该影像块进行分类"""
                x_batch_crop=x_batch[location_up:location_down,location_left:location_right] # 滑窗影像
                x_batch_crop_=np.expand_dims(x_batch_crop, axis=0)
                feed_dict = {img: x_batch_crop_,
                             phase_train:if_training
                             }
                if j==number_row-1 or i==number_col-1:
                    changeif,supp_x_batch,Box=downsampe_intpicture(x_batch_crop, out_stride)  # 符合网络大小
                    if changeif:
                        supp_x_batch=np.expand_dims(supp_x_batch,axis=0)
                        feed_dict={img: supp_x_batch,
                                   phase_train: if_training}
                pred1=sess.run(pred, feed_dict=feed_dict)
                predict=np.argmax(pred1, axis=3)
                predict=np.squeeze(predict).astype(np.uint8)

                if (j==number_row-1 or i==number_col-1) and changeif:
                    predict=image_covery(predict,Box)
                predict_concat=predict[:,result_left:result_right]  # 每个影像的预测结果，用于拼接
                if j==0:
                    Result_row=predict_concat
                else:
                    Result_row=np.hstack((Result_row,predict_concat))
                print(f"Imgae{num_img+1}:{name} ----> {i+1}/{number_col}  {j+1}/{number_row} is successfully tested ^o^  ^o^  ^o^")

            if i==0:
                Result=Result_row[result_up:result_down,:]
            else:
                predict_concat_row=Result_row[result_up:result_down,:]
                Result=np.vstack((Result,predict_concat_row))
        if not os.path.exists(save_path_predict):
            os.makedirs(save_path_predict)
        io.imsave(save_path_predict+name, Result)

with tf.Session() as sess:
    save()

total_time= get_time_dif(now_time)
print("total_time", total_time)  ##4##
