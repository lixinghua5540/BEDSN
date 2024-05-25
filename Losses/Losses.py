# -*- coding: utf-8 -*-
import  tensorflow as tf


"""
边缘检测部分的损失函数
        核心思想：将边界和非边界部分分开来进行平均，之后再将平均结果相加作为最终的损失平均值；
        其中sparse_softmax_cross_entropy_with_logits获得的为每个像素点的交叉熵值，即该点真实类别值对应的pi求得的熵值；
        则获取每类的损失熵均值，需要获取该类所有像素对应的熵值进行平均，上述tf.multiply功能实现的是获取各类对应的像素点
        labels为1代表的是边界部分；
        针对非边界部分，将(labels-1)再乘以-1时，1代表非边界，然后与loss相乘获取非边界所有loss点
        注意tf.multiply()【实现对应点相乘功能，shape不变】参数值数据类型必须保持一致，故使用tf.cast进行数据类型转换
"""

def loss_edge(labels,logits):
    label_nonedge_count=(-1)*tf.subtract(labels,1)
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    loss_p=tf.divide(tf.reduce_sum(tf.multiply(loss,tf.cast(labels,loss.dtype))),tf.cast(tf.reduce_sum(labels),dtype=tf.float32))
    loss_n=tf.divide(tf.reduce_sum(tf.multiply(loss,tf.cast(label_nonedge_count,loss.dtype))),tf.cast(tf.reduce_sum(label_nonedge_count),dtype=tf.float32))
    Loss_egde=loss_p+loss_n
    # traditional_loss=tf.reduce_mean(loss)
    return Loss_egde



"""ResUNet-a文献 所提出损失函数"""

def Tanimoto_loss(preds,label):
    loss_smooth=1.0e-5
    # Evaluate the mean volume of class per batch
    Vli=tf.reduce_mean(tf.reduce_sum(label,axis=(1,2)),axis=0)
    wli=tf.reciprocal(Vli**2)  # weighting scheme
    # First turn inf elements to zero, then replace that with the maximum weight value

    # 该两部分的目的是将权重中的inf均用inf外的最大值代替
    new_weights=tf.where(tf.is_inf(wli), tf.zeros_like(wli), wli)  # 对于满足tf.is_inf的地方取0,不满足的地方取wli,
    wli=tf.where(tf.is_inf(wli), tf.multiply(tf.ones_like(wli), tf.reduce_max(new_weights)), wli)

    rl_x_pl=tf.reduce_sum(tf.multiply(label,preds),axis=(1,2))
    # This is sum of squares
    l=tf.reduce_sum(tf.multiply(label,label),axis=(1,2))
    r=tf.reduce_sum(tf.multiply(preds,preds),axis=(1,2))
    rl_p_pl=l+r-rl_x_pl
    Tanimoto_loss=(tf.reduce_sum(tf.multiply(wli, rl_x_pl))+loss_smooth)/(tf.reduce_sum(tf.multiply(wli, (rl_p_pl)))+loss_smooth)
    return Tanimoto_loss

def Tanimoto_wth_dualloss(preds,label): # ResUNet-a提出的损失函数

    preds_dual=1.0-preds
    labels_dual=1.0-label
    loss1=Tanimoto_loss(preds,label)
    loss2=Tanimoto_loss(preds_dual,labels_dual)
    # print("loss1的值为:  {}  loss2的值为：{}".format(loss1,loss2))
    Dice_loss=(loss1+loss2)*0.5

    return Dice_loss