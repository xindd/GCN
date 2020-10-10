#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/3 15:24
# @FileName: gcn-test.py
# @Software: PyCharm
import mxnet.ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.nn import Activation

def get_dict_values(data, keys):
    return [data[k] for k in keys]

def _normal(X, eps=1e-6):
    mean = X.mean(axis=1).reshape(-1,1)
    variance = ((X-mean)**2).mean(axis=1).reshape(-1,1)
    return (X - mean) / nd.sqrt(variance + eps)


class GraphConvolution(HybridBlock):
    def __init__(self, A, in_units, out_units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        I = nd.eye(*A.shape)
        A_hat = A.copy() + I

        D = nd.sum(A_hat, axis=0)
        D_inv = D ** -0.5
        D_inv = nd.diag(D_inv)
        A_hat = nd.dot(nd.dot(D_inv, A_hat), D_inv)

        self.in_units, self.out_units = in_units, out_units

        with self.name_scope():
            self.A_hat = self.params.get_constant('A_hat', A_hat)
            self.W = self.params.get(
                'W', shape=(self.in_units, self.out_units)
            )
            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)

    def hybrid_forward(self, F, X, A_hat, W):
        # X=(S, N, M) = (样本数, 维度, 特征数)
        # print('X:', X)
        s, n, m = X.shape[0], A_hat.shape[0], A_hat.shape[0]
        w_n, w_m = W.shape[0], W.shape[1]
        A_hat = A_hat.reshape((1, n, m))
        W = W.reshape((1, w_n, w_m))
        A = F.broadcast_to(A_hat, shape=(s, n, m))
        w = F.broadcast_to(W, shape=(s, w_n, w_m))
        AX = F.batch_dot(A, X)
        # print('AX:',AX)
        # print('AXdot:',F.batch_dot(AX, w))
        propagate = self.activation(F.batch_dot(AX, w))
        # print('pro', propagate)
        return propagate


class LogisticRegressor(HybridBlock):
    def __init__(self, in_units, out_units, classnum, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(100)
            self.dense2 = nn.Dense(classnum)

    def hybrid_forward(self, F, X):
        # X=(S, N, M) = (样本数, 维度, 特征数)
        y = X.mean(axis=2, keepdims=True)
        y1, y2, y3 = y.shape
        mlp_1 = F.relu(self.dense(y.reshape((y1, y3, y2))))
        mlp_2 = self.dense2(mlp_1)
        y = F.softmax(mlp_2)
        return y


class FeaturesTransform2(HybridBlock):
    def __init__(self, entrylist, gene_to_index, entry_to_gene, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.layer_list = []
        with self.name_scope():
            for index, value in enumerate(entrylist):
                genelist = entry_to_gene[value].split(' ')
                w = self.params.get(value, shape=(1, len(genelist)))
                self.layer_list.append((get_dict_values(gene_to_index, genelist), w))

            self.entry_b = self.params.get('entry_b', shape=(len(entrylist), 1))

            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)

    def hybrid_forward(self, F, X, entry_b):
        contactlist = []
        for index, param in self.layer_list:
            contactlist.append(F.dot(param.data(), X[index, :]))
        y = nd.concat(*contactlist, dim=0) + entry_b
        return self.activation(y)


class FeaturesTransform(HybridBlock):
    def __init__(self, entrylist, gene_to_index, entry_to_gene, **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.layer_list = []
            for index, value in enumerate(entrylist):
                genelist = entry_to_gene[value].split(' ')
                # w = self.params.get(value, shape=(1, len(genelist)))
                self.layer_list.append(get_dict_values(gene_to_index, genelist))

    def hybrid_forward(self, F, X):
        S, N, M = X.shape
        res = []
        for s in range(S):
            x = X[s, :, :]
            contactlist = []
            for index in self.layer_list:
                tmp = x[index, :].sum(axis=0).reshape(1, M)
                contactlist.append(tmp)
            y = nd.concat(*contactlist, dim=0)
            res.append(y)
        re = nd.concat(*res, dim=0)
        re = re.reshape((S, len(self.layer_list), M))
        return re


class FeaturesTransform_layer(HybridBlock):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.index = index

    def hybrid_forward(self, F, X):
        # Change shape of b to comply with MXnet addition API
        # return F.dot(w, X[self.index, :])

        return X[self.index, :].mean(axis=0).reshape(1, X.shape[1])
