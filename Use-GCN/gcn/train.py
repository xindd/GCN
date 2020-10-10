#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/24 9:59
# @FileName: train.py
# @Software: PyCharm

from loadNetParams import *
from loadData import load_all_data
from models import *

from mxnet import gluon, init
from mxnet import autograd
import mxboard as mb
import time



def evaluate_accuracy(data_iter, net):
    acc, n = 0.0, 0
    for X, Y in data_iter:
        pre = net(X)
        acc += (pre.argmax(axis=1) == Y.argmax(axis=1)).sum().asscalar()
        n += Y.shape[0]
        if n == 20:
            break
    return acc / n


def train(data_iter, net, cross_entropy, trainer, num_epochs, batch_size):
    sw = mb.SummaryWriter(logdir='./logs', flush_secs=2)
    params = net.collect_params('.*W|.*dense')
    param_names = params.keys()
    ls = 0
    # train_x, train_y, test_x, test_y = allData['train_x'], allData['train_y'], allData['test_x'], allData['test_y']
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_sum, n, start = 0., 0., 0., time.time()
        for X, Y in data_iter:
            # X.attach_grad()
            with autograd.record():
                pre = net(X.reshape(*X.shape, 1))
                loss = cross_entropy(pre, Y).sum()
            loss.backward()
            trainer.step(batch_size)

            # 记录
            train_loss_sum += loss.asscalar()
            train_acc_sum += (pre.argmax(axis=1) == Y).sum().asscalar()
            n += len(Y)
            sw.add_histogram(tag='cross_entropy', values=train_loss_sum / n, global_step=ls)

            for i, name in enumerate(param_names):
                sw.add_histogram(tag=name,
                                 values=net.collect_params()[name].grad(),
                                 global_step=ls, bins=1000)
            ls += 1
        # test_acc = evaluate_accuracy(test_x, test_y, net)
        print('epoch %d, loss %.4f, train acc %.3f,  time %.1f sec' %
              (epoch + 1, train_loss_sum / n, train_acc_sum / n, time.time() - start))
    sw.close()
    return net



if __name__ == '__main__':
    dataset = LoadNetParams()
    net, feature = train_model(dataset)
    net.initialize()
    print('initialize net...')
    batch_size = 10 
    print('loading expression data...')
    exp = pd.read_csv('data/exp_data.csv')
    print('prepare train samples...')
    train_samples, validation_samples, test_samples = load_all_data()
    train_x = nd.array(exp[train_samples['sampleID']]).transpose()
    train_y = nd.array(train_samples['label'])

    dataset = gluon.data.ArrayDataset(train_x, train_y)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.trainer.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.0001})
    print('training...')
    train(data_iter, net, cross_entropy, trainer, 5, batch_size)
    print('save params...')
    net.save_parameters('data/net_params')

