# -*- coding: utf-8 -*-
# @File  : load_data.py
# @Author: ycy
# @Date  : 2019/9/7
# @Desc  :加载数据集

import numpy as np
import tensorflow as tf
dataset_path = r'C:\Users\Administrator\Desktop\data1\wisture_5ms.npz'

def get_data(path):
    npz_file = np.load(path)
    data = npz_file['data']
    labels = npz_file['labels']
    data = -data
    data -= np.mean(data,axis=0)
    data /=np.std(data,axis=0)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    labels = np.eye(3)[labels]
    #labels = np.reshape(labels, (labels.shape[0], 3))
    return data,labels


if __name__ =="__main__":
    data,label=get_data(dataset_path)
    print(data.shape)
    print(label.shape)

