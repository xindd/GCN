#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/19 9:55
# @FileName: models.py
# @Software: PyCharm

from mxnet.gluon.nn import HybridSequential
from gcn import *

def features(A, in_units, hidden_layer):
    features = HybridSequential()
    with features.name_scope():
        for i, (layer_size, activation_func) in enumerate(hidden_layer):
            layer = GraphConvolution(
                A, in_units=in_units,
                out_units=layer_size,
                activation=activation_func)
            features.add(layer)
            in_units = layer_size

    return features, in_units


def train_model(dataset):
    print('加载ppi网络数据...')
    ppi_net = dataset.get_ppi_net()
    print('加载蛋白与entry对应关系数据...')
    protein_entry_features = dataset.protein_entry_features()
    print('加载entry网络...')
    entry_net = dataset.get_entry_net()
    print('加载entry与pathway对应关系数据...')
    entry_pathway_features = dataset.entry_pathway_features()
    print('加载pathway网络数据...')
    pathway_net = dataset.get_pathway_net()
    print('构建神经网络模型...')
    genelist = dataset.genelist_order
    net = HybridSequential()
    with net.name_scope():
        ppi_in_units = len(genelist)
        # 归一化
        # net.add(nn.BatchNorm())
        # 蛋白互作
        ppi_hidden_layer = [(10, 'tanh')]  # Format: (units in layer, activation function)
        ppi_features, ppi_out_units = features(ppi_net['A'], 1, ppi_hidden_layer)
        net.add(ppi_features)
        # 归一化
        net.add(nn.BatchNorm())
        # 蛋白转entry
        ppi_to_entry_features = FeaturesTransform(protein_entry_features.entrylist,
                                                  protein_entry_features.gene_to_index,
                                                  protein_entry_features.entry_to_gene)
        net.add(ppi_to_entry_features)
        # entry互作
        entry_hidden_layer = [(10, 'tanh')]
        entry_features, entry_out_units = features(entry_net['A'], ppi_out_units, entry_hidden_layer)
        net.add(entry_features)
        # 归一化
        net.add(nn.BatchNorm())
        # entry 转 pathway
        entry_to_pathway_features = FeaturesTransform(entry_pathway_features.pathwaylist,
                                                      entry_pathway_features.entry_to_index,
                                                      entry_pathway_features.pathway_to_entry)
        net.add(entry_to_pathway_features)
        # pathway 互作
        pathway_hidden_layer = [(10, 'tanh')]
        pathway_features, pathway_out_units = features(pathway_net['A'], entry_out_units, pathway_hidden_layer)
        net.add(pathway_features)
        # 归一化
        net.add(nn.BatchNorm())
        # 分类
        classifier = LogisticRegressor(pathway_out_units, len(entry_pathway_features.pathwaylist), 33)
        net.add(classifier)
        # 归一化
        # net.add(nn.BatchNorm())
        # net.add(nn.Activation('sigmoid'))
        # classifier = LogisticRegressor(entry_out_units, len(protein_entry_features.entrylist), 2)

    # net.hybridize()

    return net, [ppi_features, entry_features, pathway_features]
