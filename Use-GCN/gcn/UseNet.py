#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/26 17:37
# @FileName: UseNet.py
# @Software: PyCharm
from models import *
from loadNetParams import *

dataset = LoadData()

net2, feature2 = train_model(dataset)

net2.load_parameters('data/net_params')
