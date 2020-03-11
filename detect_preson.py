

import sys
import argparse
from yolo import YOLO, detect_video
import numpy as np
from PIL import Image
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config=config)



class ModelBase(object):
    """
    所有模型类必须继承该类
    并且实现以下函数
    """

    def __init__(self, weight_path, use_gpu=False):
        """
        :param weight_path: 权重路径
        :param use_gpu:     是否使用GPU
        """
        self.weight_path = weight_path
        self.use_gpu = use_gpu
        self.net = None

    def pre_img(self, image):
        """
        预处理图片
        :param image: numpy.ndarray 图片
        :return:
        """
        raise NotImplementedError

    def load_model(self):
        """
        初始化模型(给self.net赋值)
        """
        raise NotImplementedError

    def predict(self, image_info):
        """
        对图片进行预测
        :param image_info:
        :return:
        """
        raise NotImplementedError


class Yolov3(ModelBase):

    def __init__(self):
        """
        :param weight_path: 权重路径
        :param use_gpu:     是否使用GPU
        """
        # self.weight_path = weight_path
        # self.use_gpu = use_gpu
        # self.net = None
        self.yolo = YOLO()

    def predict(self,imagepath):
        """
        对图片进行预测
        :param image_info:
        :return:
        """
        # filenames = os.listdir(inputpath)
        # for i, file in enumerate(filenames):
        # image_info = {}
        savepath = './output/'
        try:
            image = Image.open(imagepath)
        except:
            print('Open Error! Try again!')
        else:
            r_image, time ,walker_bbox = self.yolo.detect_image(image)
            time = round(time, 3)
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            r_image.save(savepath  + '_' + str(time) + '.jpg')
        # image = np.array(image, dtype='float32')
        # image_info['image'] = image
        # image_info['walker_bbox'] = walker_bbox
        # return image_info
        self.yolo.close_session()

if __name__ == "__main__":
    inputpath = './test1.jpg'

    pre = Yolov3()
    image_info = pre.predict(inputpath)
