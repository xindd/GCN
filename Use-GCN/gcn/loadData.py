#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/29 15:49
# @FileName: loadData.py
# @Software: PyCharm

import pandas as pd
import random
from imblearn.over_sampling import RandomOverSampler
import os
import numpy as np

def load_all_data_1():
    root = 'data/raw_exp'
    file_list = os.listdir(root)
    file_labels = dict(zip(file_list, list(range(1, len(file_list) + 1))))
    samples_list = []

    for file in file_list:
        with open(os.path.join(root, file), 'r') as f:
            first_row = f.readline()
            first_row = first_row.strip().split('\t')
            for sample in first_row[1:]:
                if int(sample.strip()[-3:-2]) == 0:
                    # case
                    samples_list.append((sample, file_labels[file]))
                else:
                    # control
                    samples_list.append((sample, 0))
    sample2labels = pd.DataFrame(samples_list, columns=['sampleID', 'label'])
    ros = RD(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(np.array(sample2labels['sampleID']).reshape(-1, 1),
                                              np.array(sample2labels['label'])
                                              )  # 过采样
    allsamples = pd.DataFrame({'sampleID': X_resampled[:, 0], 'label': y_resampled})
    length = allsamples.shape[0]
    index_list = list(range(length))
    train_num, validation_num, test_num = 0.7, 0.2, 0.1
    random.shuffle(index_list)
    train_index = index_list[0:int(length * train_num)]
    validation_index = index_list[int(length * train_num): int(length * train_num) + int(length * validation_num)]
    test_index = index_list[int(length * train_num) + int(length * validation_num):]

    train_samples = allsamples.loc[train_index]
    validation_samples = allsamples.loc[validation_index]
    test_samples = allsamples.loc[test_index]

    return train_samples, validation_samples, test_samples

def processSample():
    root = 'data/raw_exp'
    file_list = os.listdir(root)
    file_labels = dict(zip(file_list, list(range(1,len(file_list)+1))))
    samples_list = []
    
    for file in file_list:
        with open(os.path.join(root,file), 'r') as f:
            first_row = f.readline()
            first_row = first_row.strip().split('\t')
            for sample in first_row[1:]:
                if int(sample.strip()[-3:-2]) == 0:
                    # case
                    samples_list.append((sample, file_labels[file]))
                else:
                    # control
                    samples_list.append((sample, 0))
    m = np.array(samples_list)
    np.save('data/sampleList.npy',m)

def load_all_data(reload=False):
    if reload:
        processSample()
    a = np.load('data/sampleList.npy')
    samples_list=a.tolist()
    
    sample2labels = pd.DataFrame(samples_list,columns=['sampleID', 'label'])
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_sample(np.array(sample2labels['sampleID']).reshape(-1, 1), 
                                              np.array(sample2labels['label'])
                                             ) # 过采样
    allsamples = pd.DataFrame({'sampleID':X_resampled[:,0], 'label':y_resampled})
    length = allsamples.shape[0]
    index_list = list(range(length))
    train_num, validation_num, test_num= 0.7, 0.2, 0.1
    random.shuffle(index_list)
    train_index = index_list[0:int(length*train_num)]
    validation_index = index_list[int(length*train_num): int(length*train_num) + int(length*validation_num)]
    test_index = index_list[int(length * train_num) + int(length * validation_num):]

    train_samples = allsamples.loc[train_index]
    validation_samples = allsamples.loc[validation_index]
    test_samples = allsamples.loc[test_index]
    
    return train_samples, validation_samples, test_samples

