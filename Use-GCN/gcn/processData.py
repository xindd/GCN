#!/usr/bin/python3
# -*- coding:utf-8 -*-
# @Author: XinWang
# @Time:2019/7/19 10:09
# @FileName: processData.py
# @Software: PyCharm

from Bio.KEGG.REST import *
from Bio.KEGG.KGML import KGML_parser
import pandas as pd
from io import StringIO
import pickle

pd.set_option('display.max_colwidth',500) # 设置DataFrame显示的宽度
# 提取 pathway，保存 pathway 节点
def extract_pathways_nodes(species='hsa', path='data/', save=True):
    print('Executing function extract_pathways_nodes:')
    pathway_nodes = kegg_list('pathway', species).read()
    pathway_nodes_df = pd.read_csv(StringIO(pathway_nodes), sep='\t', header=None)
    if save:
        print('--- save pathway nodes to %s ...' % (path + 'pathway_nodes.csv'))
        pathway_nodes_df[0].to_csv(path + 'pathway_nodes.csv', header=0, index=0)
        print('--- save pathway annotate to %s ...' % (path + 'pathway_nodes_annotate.csv'))
        pathway_nodes_df.to_csv(path + 'pathway_nodes_annotate.csv', header=0, index=0)
    return pathway_nodes_df


# 解析 xml 文件
# 1. 提取 pathway--pathway 关系
# 2. 提取每个 pathway 下 entry_nodes
#    A. 提取 entry--entry 关系
#    B. 提取 entry--gene 对应关系
#    C. 提取 entry nodes
def parser_xml(pathway_nodes_df, save=True):
    print('Executing function parser_xml:')
    pathwaylist = list(pathway_nodes_df[0])  # 获取 pathway 列表
    enzymeDict = {}  # 初始化结果字典
    path2path = list()  # 初始化
    procress, allnum = 0, len(pathwaylist)  # 显示进度

    for pathwayname in pathwaylist:
        if procress % (allnum // 5) == 0:
            print(f"--- Parsing xml {procress}/{allnum} pathwayname: {pathwayname}")
        procress += 1
        # 在线提取 pathwayname 下的 xml 文件
        pathway2xml = KGML_parser.read(kegg_get(pathwayname, "kgml"))
        # 将 xml 内出现的所有 map 类型保存，并认为这些 map 与 pathwayname 有互作关系
        path2path.extend([[pathwayname, maps.name] for maps in pathway2xml.maps if maps.name in pathwaylist])
        # 设置空的 dataframe 存储 pathwayname 下 entry 与 entry 关系
        relation2entry = pd.DataFrame(columns=('id1', 'id2'))
        genelist = pathway2xml.genes
        for i, gene in enumerate(pathway2xml.relations):
            if gene.entry1 in genelist and gene.entry2 in genelist:
                relation2entry.loc[i, :] = [pathwayname + '_' + str(gene.entry1.id),
                                            pathwayname + '_' + str(gene.entry2.id)]

        entry_nodes = list(set(relation2entry['id1'].tolist() + relation2entry['id2'].tolist()))
        id2gene = [(pathwayname + '_' + str(gene.id), gene.name) for gene in genelist]
        enzymeDict[pathwayname] = {'entry_nodes': entry_nodes, 'entry_entry_edges': relation2entry,
                                   'entry2gene': id2gene}
    print('--- Finish parsing xml')
    print('--- Processing data...')
    # 去重
    path2path = pd.DataFrame(path2path, columns=('path1', 'path2'))
    rows = [i for i in path2path.index if path2path.iat[i, 0] == path2path.iat[i, 1]]
    path2path2 = path2path.drop(rows, axis=0)  # 利用drop方法将含 path1=path2 的行删除
    path2path2 = path2path2.drop_duplicates(['path1', 'path2'], keep='first')  # 删除重复行
    a = path2path2.apply(lambda x: str(sorted(x.tolist())), axis=1)
    pos = pd.DataFrame(a).duplicated()
    path2path_drop = path2path2.loc[-pos, :].reset_index()
    # 保存
    if save:
        print('--- Saving data pathway2enzyme.pickle.txt')
        with open('data/pathway2enzyme.pickle.txt', 'wb') as file:
            pickle.dump(enzymeDict, file)
        print('--- Saving data pathway2pathway.csv')
        path2path_drop[['path1', 'path2']].to_csv('data/pathway2pathway.csv', header=0, index=0)
    return {'pathway2enzyme': enzymeDict, 'pathway2pathway': path2path_drop}


# 将 hsa 与 gene symbol 对应
def hsa2symbol(hgncPath, ppiPath, enzymeDict, save=True):
    print('Executing function hsa2symbol:')
    allgene = []
    for path in list(enzymeDict.keys()):
        for gene in enzymeDict[path]['entry2gene']:
            allgene.extend(gene[1].split(' '))
    allgene_drop = list(set(allgene))
    print('--- hsa numbers: %d' % (len(allgene_drop)))
    # hgncPath = 'E:/wx/2019上课题/数据/hgnc_complete_set.txt'
    hgncFile = pd.read_csv(hgncPath, sep='\t',
                           usecols=['hgnc_id', 'symbol', 'name', 'alias_symbol', 'entrez_id',
                                    'ensembl_gene_id', 'uniprot_ids', 'enzyme_id'],
                           engine='python',
                           dtype={'entrez_id': str})
    gene2symbol = hgncFile.loc[hgncFile['entrez_id'].isin([i[4:] for i in allgene_drop])]
    # ppiPath = 'E:/wx/2019上课题/数据/9606.protein.info.v11.0.txt'
    PPI_gene = pd.read_csv(ppiPath, sep='\t',
                           usecols=['protein_external_id', 'preferred_name'],
                           engine='python')

    # 将 xml 提取的 gene 与 蛋白互作数据中的 gene 取交集
    finalgene = PPI_gene.loc[PPI_gene['preferred_name'].isin(gene2symbol['symbol'])]
    print(f'--- protein & gene numbers {finalgene.shape[0]}')
    # 将 gene 与表达谱中的 gene 交集
    symbol2ensg = hgncFile.loc[hgncFile['symbol'].isin(finalgene['preferred_name'])]
    ensemblid = pd.read_csv('E:/wx/2019上课题/数据/exp/TCGA-ACC.htseq_fpkm-uq.tsv', sep='\t', usecols=['Ensembl_ID'],
                            engine='python')
    ensemblid['id'] = ensemblid['Ensembl_ID'].apply(lambda x: x.split('.')[0])
    symbol2ensg = symbol2ensg.drop_duplicates(['ensembl_gene_id'], keep=False)
    enseid_list = set(symbol2ensg['ensembl_gene_id']) & set(ensemblid['id'])
    symbol2ensg = symbol2ensg.loc[symbol2ensg['ensembl_gene_id'].isin(enseid_list)]
    symbol2ensg['ensembl_gene_version'] = symbol2ensg['ensembl_gene_id'].map(
        dict(zip(ensemblid['id'], ensemblid['Ensembl_ID'])))
    print(f'--- gene & exp numbers {symbol2ensg.shape[0]}')
    # 提取 hsa:xxx gene 与 symbol 对应关系
    hsa2symbol = hgncFile.loc[hgncFile['symbol'].isin(symbol2ensg['symbol'])][['symbol', 'entrez_id']]
    hsa2symbol['entrez_id'] = hsa2symbol.apply(lambda x: 'hsa:' + str(x[1]), axis=1)
    hsa2symbol.reset_index(drop=True)
    print(f'--- protein numbers {hsa2symbol.shape[0]}')
    finalgene = finalgene.loc[finalgene['preferred_name'].isin(symbol2ensg['symbol'])]
    if save:
        print('--- Saving data symbol_nodes.csv')
        finalgene['preferred_name'].to_csv('data/symbol_nodes.csv', header=0, index=0)
        print('--- Saving data hsa2symbol.csv')
        hsa2symbol.to_csv('data/hsa2symbol.csv', header=0, index=0)
        print('--- Saving data symbol2protein.csv')
        finalgene.to_csv('data/symbol2protein.csv', header=0, index=0)
    return {'symbol': finalgene['preferred_name'], 'hsa2symbol': hsa2symbol, 'symbol2protein': finalgene}

# 利用 protein 互作网络提取 symbol 互作网络
def ppi2symbols(ppiPath, savePath, symbol2protein):
    print('Executing function ppi2symbols:')
    protein_ensp_id = list(symbol2protein['protein_external_id'])
    proetin_protein = []
    # savePath = 'data/protein_protein_edges.csv'
    outputfile = open(savePath, 'w')
    # ppiPath = 'E:/wx/2019上课题/数据/9606.protein.links.v11.0.txt'
    print('--- Running, this may take a long time, please be patient')
    with open(ppiPath, 'r') as file:
        i = 0
        for r in file.readlines():
            if i % 500000==0:
                print(f'--- {i} lines that have been read')
            i += 1
            p1, p2, s = r.strip().split(' ')
            if p1 in protein_ensp_id and p2 in protein_ensp_id:
                outputfile.write(symbol2protein.loc[symbol2protein.protein_external_id==p1, 'preferred_name'].values[0] + ',' +
                                 symbol2protein.loc[symbol2protein.protein_external_id==p2, 'preferred_name'].values[0] + ',' + s + '\n')
    outputfile.close()
    print(f'--- Finish, save the file to {savePath}')
    return

# 对 symbol 互作网络去重
def ppi2duplicated(ppiPath):
    print('Executing function ppi2duplicated:')
    #ppiPath = 'data/protein_protein_edges.csv'
    PPI = pd.read_csv(ppiPath, sep=',',
                      names =['p1','p2', 'score'],
                      header=None,
                      engine='python')
    sortppi = PPI[['p1', 'p2']].apply(lambda x: str(sorted(x.tolist())), axis=1)
    pos = pd.DataFrame(sortppi).duplicated()
    ppi_drop = PPI.loc[-pos,:].reset_index()
    ppi_drop = ppi_drop[['p1','p2', 'score']]
    # savePath = 'data/protein_protein_edges_drop_duplicated.csv'
    print('--- Saving data protein_protein_edges.csv')
    ppi_drop.to_csv('data/protein_protein_edges.csv', header=0, index=0)
    return

if __name__ == '__main__':
    pathwaylist = extract_pathways_nodes()
    parser_res=parser_xml(pathway_nodes_df=pathwaylist)

    hgncPath = 'E:/wx/2019上课题/数据/hgnc_complete_set.txt'
    ppiPath = 'E:/wx/2019上课题/数据/9606.protein.info.v11.0.txt'
    res = hsa2symbol(hgncPath=hgncPath, ppiPath=ppiPath, enzymeDict=parser_res['pathway2enzyme'])

    savePath = 'data/protein_protein_edges_duplicated.csv'
    ppiPath = 'E:/wx/2019上课题/数据/9606.protein.links.v11.0.txt'
    ppi2symbols(ppiPath, savePath, symbol2protein=res['symbol2protein'])

    ppiPath = 'data/protein_protein_edges_duplicated.csv'
    ppi2duplicated(ppiPath)
