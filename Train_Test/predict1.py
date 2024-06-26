# -*- coding: utf-8 -*-
"""用于小影像测试集的预测"""
import tensorflow as tf

from Semantic_Segmentation.Models.SCAttNet.SCAttNet import inference

import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import scipy
import warnings
warnings.filterwarnings("ignore")

from skimage import io

batch_size = 1
img = tf.placeholder(tf.float32, [batch_size, 512,512,3])
test_img = sorted(glob.glob("XXXX/test_images/*.png"))
phase_train = tf.placeholder(tf.bool, name='phase_train')
pred = inference(img, phase_train)
saver = tf.train.Saver()


def save():
    tf.global_variables_initializer().run()
    checkpoint_dir = 'checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    for j in range(0, len(test_img)):
        print(j,"^o^ ^o^")
        x_batch = test_img[j]
        i = x_batch.split('/')[-1]
        name=os.path.basename(x_batch)
        x_batch = io.imread(x_batch) / 255.0
        x_batch = np.expand_dims(x_batch, axis=0)
        feed_dict = {img: x_batch,
                              phase_train: False }
        pred1 = sess.run(pred, feed_dict=feed_dict)
        predict = np.argmax(pred1, axis=3)
        predict = np.squeeze(predict).astype(np.uint8)
        io.imsave('./Results/'+name, predict)


def vis():
    tf.global_variables_initializer().run()
    checkpoint_dir = 'checkpoint/'
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    for j in range(0, len(test_img)):
        x_batch = test_img[j]
        x_batch1 = scipy.misc.imread(x_batch) / 255.0
        x_batch = np.expand_dims(x_batch1, axis=0)
        feed_dict = {img: x_batch,
                     phase_train: False

                     }
        pred1 = sess.run(pred, feed_dict=feed_dict)
        predict = np.argmax(pred1, axis=3)
        predict = np.squeeze(predict)

        plt.imshow(x_batch1)
        plt.show()
        plt.imshow(predict)
        plt.show()


with tf.Session() as sess:
    save()