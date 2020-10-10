#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/25 16:30
# @FileName: t.py
# @Software: PyCharm
from collections import namedtuple
import pickle
import pandas as pd
import networkx as nx
from mxnet import ndarray as nd


# 加载 gene entry 对对应数据
# genelist_order, gene_to_index, entrylist, entry_to_gene = load_gene_to_entry()
# 加载 gene entry 对对应数据
# 加载 gene entry 对对应数据
def load_gene_to_entry():
    hsa2symbol = pd.read_csv('data/hsa2symbol.csv', sep=',', header=None)
    hsa2symbol = hsa2symbol.set_index(hsa2symbol[1])

    # gene_to_index
    genelist = pd.read_csv('data/symbol_nodes.csv', sep='\t', header=None)
    genelist_order = sorted(list(genelist[0]))
    gene_to_index = dict(zip(genelist_order, list(range(len(genelist_order)))))
    with open('data/pathway2enzyme.pickle.txt', 'rb') as file:
        enzymeDict_load = pickle.load(file)
    res = [i for path in enzymeDict_load.keys() for i in enzymeDict_load[path]['entry2gene']]
    res = pd.DataFrame(res)
    for i in range(res.shape[0]):
        gl = [hsa2symbol.at[h, 0] for h in res.iat[i, 1].split(' ') if h in hsa2symbol.index]
        res.iat[i, 1] = ' '.join(gl)
    res = res.loc[-(res[1] == '')]

    # entrylist,
    entrygene = res.loc[-res[1].duplicated(), 1]
    entrygene = entrygene.reset_index(drop=True).reset_index()
    geneset2entryID = dict(zip(entrygene[1], entrygene['index']))
    res['entryID'] = res[1].map(geneset2entryID).apply(lambda x: 'entry' + str(x))
    entrylist = sorted(list(set(res['entryID'])), key=lambda x: int(x[5:]))

    # entry_to_gene
    entry_to_gene = dict(zip(res['entryID'], res[1]))

    # pathwaylist
    pathway_nodes = pd.read_csv('data/pathway_nodes.csv', header=None)
    pathwaylist = sorted(list(pathway_nodes[0]))

    # pathway_to_entry
    res['pathwayID'] = res[0].apply(lambda x: x.split('_')[0])
    pathway_to_entry = dict(res.groupby('pathwayID')['entryID'].apply(lambda x: ' '.join(x)))

    # entry_to_index
    entry_to_index = dict(zip(entrylist, list(range(len(entrylist)))))

    FeaturesTransformSet = namedtuple(
        'FeaturesTransformSet',
        field_names=['genelist_order', 'gene_to_index', 'entrylist', 'entry_to_gene',
                     'pathwaylist', 'entry_to_index', 'pathway_to_entry'])

    return FeaturesTransformSet(genelist_order, gene_to_index, entrylist, entry_to_gene,
                                pathwaylist, entry_to_index, pathway_to_entry)


def get_entry_net():
    hsa2symbol = pd.read_csv('data/hsa2symbol.csv', sep=',', header=None)
    hsa2symbol = hsa2symbol.set_index(hsa2symbol[1])

    genelist = pd.read_csv('data/symbol_nodes.csv', sep='\t', header=None)
    genelist_order = sorted(list(genelist[0]))
    with open('data/pathway2enzyme.pickle.txt', 'rb') as file:
        enzymeDict_load = pickle.load(file)
    res = [i for path in enzymeDict_load.keys() for i in enzymeDict_load[path]['entry2gene']]
    res = pd.DataFrame(res)
    for i in range(res.shape[0]):
        gl = [hsa2symbol.at[h, 0] for h in res.iat[i, 1].split(' ') if h in hsa2symbol.index]
        res.iat[i, 1] = ' '.join(gl)
    res = res.loc[-(res[1] == '')]
    # entrylist,
    entrygene = res.loc[-res[1].duplicated(), 1]
    entrygene = entrygene.reset_index(drop=True).reset_index()
    geneset2entryID = dict(zip(entrygene[1], entrygene['index']))
    res['entryID'] = res[1].map(geneset2entryID).apply(lambda x: 'entry' + str(x))

    tmpdict = dict(zip(res[0], res['entryID']))
    entry_to_entry = pd.DataFrame(columns=['id1', 'id2'])
    for k in enzymeDict_load.keys():
        entry_tmp = pd.DataFrame(columns=['id1', 'id2'])
        entry_tmp['id1'] = enzymeDict_load[k]['entry_entry_edges']['id1'].map(tmpdict)
        entry_tmp['id2'] = enzymeDict_load[k]['entry_entry_edges']['id2'].map(tmpdict)
        entry_tmp = entry_tmp.dropna().reset_index(drop=True)
        entry_to_entry = pd.concat([entry_to_entry, entry_tmp])
    entry_to_entry_drop = entry_to_entry.drop_duplicates()
    entry_to_entry_drop = entry_to_entry_drop.loc[-(entry_to_entry_drop['id1'] == entry_to_entry_drop['id2'])]
    a = entry_to_entry_drop.apply(lambda x: str(sorted(x.tolist())), axis=1)
    pos = pd.DataFrame(a).duplicated()
    entry_to_entry_drop = entry_to_entry_drop.loc[-pos].reset_index(drop=True)

    entry_net_nx = nx.from_pandas_edgelist(entry_to_entry_drop, 'id1', 'id2')
    entry_A = nd.array(nx.to_numpy_matrix(entry_net_nx))

    return {'edges': entry_to_entry_drop, 'A': entry_A}


def get_ppi_net():
    # 加载gene顺序
    genelist = pd.read_csv('data/symbol_nodes.csv', sep='\t', header=None)
    genelist_order = sorted(list(genelist[0]))
    # 加载PPI
    ppi_net = pd.read_csv('data/protein_protein_edges.csv', sep=',', names=['p1', 'p2', 's'], header=None)
    ppi_net_nx = nx.from_pandas_edgelist(ppi_net, 'p1', 'p2')
    ppi_A = nd.array(nx.to_numpy_matrix(ppi_net_nx, nodelist=genelist_order))
    return {'edges': ppi_net, 'A': ppi_A}

def get_pathway_net():
    pathway_net = pd.read_csv('data/pathway2pathway.csv', sep=',', names=['p1', 'p2'])
    pathway_net_nx = nx.from_pandas_edgelist(pathway_net, 'p1', 'p2')
    pathway_A = nd.array(nx.to_numpy_matrix(pathway_net_nx))
    return {'edges': pathway_net, 'A': pathway_A}


class FeaturesTransform(HybridBlock):
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
        # Change shape of b to comply with MXnet addition API
        contactlist = []
        for index, param in self.layer_list:
            contactlist.append(F.dot(param.data(), X[index, :]))
        y = nd.concat(*contactlist, dim=0) + entry_b
        return self.activation(y)


#gcn.py
#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/3 15:24
# @FileName: gcn-test.py
# @Software: PyCharm
import mxnet.ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.nn import Activation


class GraphConvolution(HybridBlock):
    def __init__(self, A, in_units, out_units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        I = nd.eye(*A.shape)
        A_hat = A.copy() + I

        D = nd.sum(A_hat, axis=0)
        D_inv = D ** -0.5
        D_inv = nd.diag(D_inv)

        A_hat = D_inv * A_hat * D_inv

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
        aggregate = F.dot(A_hat, X)
        propagate = self.activation(
            F.dot(aggregate, W))
        return propagate


class LogisticRegressor(HybridBlock):
    def __init__(self, in_units, out_units, classnum,**kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get('w', shape=(in_units, 1))
            self.b = self.params.get('b', shape=(out_units, 1))
            self.dense = nn.Dense(100)
            self.dense2 = nn.Dense(classnum)

    def hybrid_forward(self, F, X, w, b):
        # Change shape of b to comply with MXnet addition API
        # b = F.broadcast_axis(b, axis=(0,1), size=(333, 1))
        y = F.dot(X, w) + b
        mlp_1 = F.sigmoid(self.dense(y.reshape(1, y.shape[0])))
        mlp_2 = F.sigmoid(self.dense2(mlp_1))
        return mlp_2


def get_dict_values(data, keys):
    return [data[k] for k in keys]


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
        #         print(layer_list)
        # Change shape of b to comply with MXnet addition API
        contactlist = []
        for index, param in self.layer_list:
            contactlist.append(F.dot(param.data(), X[index, :]))
        y = nd.concat(*contactlist, dim=0) + entry_b
        return self.activation(y)


class FeaturesTransform(HybridBlock):
    def __init__(self, entrylist, gene_to_index, entry_to_gene, activation='relu', **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.layer_list = nn.HybridSequential()
            for index, value in enumerate(entrylist):
                genelist = entry_to_gene[value].split(' ')
                # w = self.params.get(value, shape=(1, len(genelist)))
                self.layer_list.add(FeaturesTransform_layer(index=get_dict_values(gene_to_index, genelist),
                                                            value=value,
                                                            length=len(genelist)))
            self.entry_b = self.params.get('entry_b', shape=(len(entrylist), 1))

            if activation == 'identity':
                self.activation = lambda X: X
            else:
                self.activation = Activation(activation)

    def hybrid_forward(self, F, X, entry_b):
        # Change shape of b to comply with MXnet addition API
        contactlist = []
        for layer in self.layer_list:
            contactlist.append(layer(X))
        y = nd.concat(*contactlist, dim=0) + entry_b
        return self.activation(y)


class FeaturesTransform_layer(HybridBlock):
    def __init__(self, index, value, length, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.w = self.params.get(value, shape=(1, length))
            self.index = index

    def hybrid_forward(self, F, X, w):
        # Change shape of b to comply with MXnet addition API
        return F.dot(w, X[self.index, :])


class FeaturesTransform(HybridBlock):
    def __init__(self, entrylist, gene_to_index, entry_to_gene, **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.layer_list = nn.HybridSequential()
            for index, value in enumerate(entrylist):
                genelist = entry_to_gene[value].split(' ')
                # w = self.params.get(value, shape=(1, len(genelist)))
                self.layer_list.add(FeaturesTransform_layer(index=get_dict_values(gene_to_index, genelist)))

    def hybrid_forward(self, F, X):
        # Change shape of b to comply with MXnet addition API
        contactlist = []
        for layer in self.layer_list:
            contactlist.append(layer(X))
        y = nd.concat(*contactlist, dim=0)
        return y


class FeaturesTransform_layer(HybridBlock):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.index = index

    def hybrid_forward(self, F, X):
        # Change shape of b to comply with MXnet addition API
        # return F.dot(w, X[self.index, :])
        return X[self.index, :].mean(axis=0)