#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/19 9:56
# @FileName: train.py
# @Software: PyCharm

from mxnet.gluon.loss import SigmoidBinaryCrossEntropyLoss
from mxnet.gluon.trainer import Trainer
import mxnet.ndarray as nd
from mxnet import autograd
from networkx import to_numpy_matrix, shortest_path_length
from sklearn.metrics import classification_report

from loadNetParams import load_karate_club
from models import build_model, calc_accuracy
import matplotlib.pyplot as plt
import numpy as np



def showData(output, G):
    for i in range (34):
        if G.nodes[i]['role'] == 'Administrator':
            plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,label=str(i),color = 'green',alpha=0.8,s = 250)
            plt.text(np.array(output)[i,0],np.array(output)[i,1] ,i, horizontalalignment='center',verticalalignment='center', fontdict={'color':'black'})
        elif G.nodes[i]['role'] == 'Instructor':
            plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,label=str(i),color = 'green',alpha=0.8,s = 250)
            plt.text(np.array(output)[i,0],np.array(output)[i,1] ,i, horizontalalignment='center',verticalalignment='center', fontdict={'color':'black'})
        elif G.nodes[i]['community'] == 'Administrator':
            plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,label=str(i),color = 'b',alpha=0.2,s = 250)
            plt.text(np.array(output)[i,0],np.array(output)[i,1] ,i, horizontalalignment='center',verticalalignment='center', fontdict={'color':'black'})
            # 为每个点添加标签，一些形如（x轴，y轴，标签）的元组，水平及垂直位置，背景颜色
        else:
            plt.scatter(np.array(output)[i,0],np.array(output)[i,1] ,label = 'i',color = 'r',alpha=0.2,s = 250)
            plt.text(np.array(output)[i,0],np.array(output)[i,1] ,i, horizontalalignment='center',verticalalignment='center', fontdict={'color':'black'})
            # plt.scatter(np.array(output)[:,0],np.array(output)[:,1],label = 0:33)

def train(model, features, X, X_train, y_train, epochs):
    cross_entropy = SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.001, 'momentum': 1})

    feature_representations = [features(X).asnumpy()]
    plt.figure()
    for e in range(1, epochs + 1):
        cum_loss = 0
        cum_preds = []

        for i, x in enumerate(X_train):
            y = nd.array(y_train)[i]
            with autograd.record():
                preds = model(X)[x]
                # print(model(X).shape)
                # print(x)
                loss = cross_entropy(preds, y)
            loss.backward()
            trainer.step(1)

            cum_loss += loss.asscalar()
            cum_preds += [preds.asscalar()]
        plt.cla()
        plt.title('epochs'+str(e))
        showData(features(X).asnumpy(), zkc.network)
        plt.pause(0.001)
        feature_representations.append(features(X).asnumpy())

        if (e % (epochs // 10)) == 0:
            print(f"Epoch {e}/{epochs} -- Loss: {cum_loss: .4f}")
            print(cum_preds)
    plt.show()
    return feature_representations

zkc = load_karate_club()

A = to_numpy_matrix(zkc.network)
A = nd.array(A)
X_train = zkc.X_train.flatten()
y_train = zkc.y_train
X_test = zkc.X_test.flatten()
y_test = zkc.y_test

X_1 = I = nd.eye(*A.shape)
model_1, features_1 = build_model(A, X_1)
# print(model_1(X_1))
# feature_representations_1 = train(model_1, features_1, X_1, X_train, y_train, epochs=5000)
# y_pred_1 = calc_accuracy(model_1, X_1, X_test)
# print(classification_report(y_test, y_pred_1))
# plt.figure()
# for i in range(250):
#     res = feature_representations_1[i]
#     plt.cla()
#     showData(res, zkc.network)
#     plt.pause(0.01)
#
# plt.show()
X_2 = nd.zeros((A.shape[0], 2))
node_distance_instructor = shortest_path_length(zkc.network, target=33)
node_distance_administrator = shortest_path_length(zkc.network, target=0)

for node in zkc.network.nodes():
    X_2[node][0] = node_distance_administrator[node]
    X_2[node][1] = node_distance_instructor[node]
X_2 = nd.concat(X_1, X_2)
model_2, features_2 = build_model(A, X_2)
feature_representations_2= train(model_2, features_2, X_2, X_train, y_train, epochs=1000)