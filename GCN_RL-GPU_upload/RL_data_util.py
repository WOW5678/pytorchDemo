# -*- coding: utf-8 -*-
"""
 @Time    : 2019/3/20 0020 下午 8:33
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function: 按照GCN需要的格式 准备好邻接矩阵
"""
import numpy as np
import  scipy.sparse as sp
import os
import sys
import pickle as pkl
from collections import Counter
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch

np.random.seed(1001)

def init_ADJ(nodes_dict):
    adjacencies = []
    adj_shape = (len(nodes_dict), len(nodes_dict))

    edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
    for j in range(len(nodes_dict)):
        edges[j] = np.array([j, j])
    row, col = np.transpose(edges)
    data = np.zeros(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_combination.npz',adj)
    adjacencies.append(adj)
    adjacencies.append(adj)
    # 处理对抗关系 生成对抗元组的邻接矩阵(初始时也全部为0)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_adverse.npz', adj)
    del edges, adj,data,row,col
    return adjacencies


def getADJ(action,seletectedActions,nodes_dict):
    adjacencies=[]
    adj_shape=(len(nodes_dict),len(nodes_dict))

    spoList_rel0,spoList_rel1=get_spo(action, seletectedActions, nodes_dict)
    edges = np.empty((len(spoList_rel0), 2), dtype=np.int32)
    for j,(s,p,o) in enumerate(spoList_rel0):
        #print('s-p-o:',(s,p,o))
        edges[j]=np.array([s,o])
    row,col=np.transpose(edges)
    data=np.ones(len(row))
    adj=sp.csr_matrix((data,(row,col)),shape=adj_shape,dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_combination.npz',adj)
    adjacencies.append(adj)

    #处理对抗关系 生成对抗元组的邻接矩阵
    if len(spoList_rel1)==0:
        adj_shape = (len(nodes_dict), len(nodes_dict))

        edges = np.empty((len(nodes_dict), 2), dtype=np.int32)
        for j in range(len(nodes_dict)):
            edges[j] = np.array([j, j])
        row, col = np.transpose(edges)
        data = np.zeros(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    else:
        edges = np.empty((len(spoList_rel1), 2), dtype=np.int32)
        for j, (s, p, o) in enumerate(spoList_rel1):
            #print('s-p-o:',(s,p,o))
            edges[j] = np.array([s, o])
        row, col = np.transpose(edges)
        data = np.ones(len(row))
        adj = sp.csr_matrix((data, (row, col)), shape=adj_shape, dtype=np.uint8)
    #save_sparse_csr('/home/wangshanshan/GCN_RL/adjData/rel_adverse.npz', adj)
    adjacencies.append(adj)
    #print('adjacencies:',adjacencies)
    del adj,edges
    return adjacencies

def get_spo(action,seletectedActions,nodes_dict):
    spoList0=[]
    #先添加组合药物组
    spoList0.append([action,'组合',action])
    if len(seletectedActions)!=0:
        for id in seletectedActions:
            spoList0.append([action,'组合',id])
            spoList0.append([id,'组合',action])

    spoList1=[]
    #在从DDI药物对种选择出与该action相关的对抗药物 添加到对抗药物组中
    with open('/home/wangshanshan/GCN_RL-GPU/data/newDDIData_filter.csv','r',newline='',encoding='utf-8') as f:
        # print('nodes_dict:',nodes_dict)
        # print('len(nodes_dict):',len(nodes_dict))
        ddiData=[]
        reader=csv.reader(f)
        for row in reader:
            if nodes_dict.get(row[0])==None:
                print('row[0]:',row[0])
            ddiData.append([nodes_dict.get(row[0]),nodes_dict.get(row[1])])

        for row in ddiData:
            if action==row[0]:
                print('action-对抗-row[2]:',[action,'对抗',row[1]])
                spoList1.append([action,'对抗',row[1]])
            if action==row[1]:
                spoList1.append([row[0],'对抗',action])
    del ddiData
    return spoList0,spoList1

def save_sparse_csr(filename,array):
    np.savez(filename,data=array.data,indices=array.indices,indptr=array.indptr,shape=array.shape)

def load_data(filename):
    patients=[]
    drugs=[]
    drugSet=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            drugL=row[6].split(' ')
            #row[4]为病人描述
            row[4]=row[4].replace(' ','')
            tmpL=[item for item in drugL if item]
            if len(tmpL)>0:
                patients.append(' '.join(row[4]))
                drugs.append(tmpL)
                drugSet.extend(tmpL)
    return patients,drugs,list(set(drugSet))

def load_data_diagnosis(filename):
    patients=[]
    drugs=[]
    drugSet=[]
    with open(filename,'r',encoding='utf-8') as f:
        reader=csv.reader(f)
        for row in reader:
            drugL=row[6].split(' ')
            #row[5]为病人诊断结果
            row[5]=row[5].replace(' ','')
            tmpL=[item for item in drugL if item]
            if len(tmpL)>0:
                patients.append(' '.join(row[5]))
                drugs.append(tmpL)
                drugSet.extend(tmpL)

    return patients,drugs,list(set(drugSet))


def tokenization(patients):

    #计算出patients序列中最长的样本
    maxLength=max([len(line.split(' ')) for line in patients])
    print('maxLength:',maxLength)
    #序列化samples
    patients_tokenizer = Tokenizer()
    #print('sample:',samples)
    patients_tokenizer.fit_on_texts(patients)
    sequences = patients_tokenizer.texts_to_sequences(patients)
    #print(sequences)
    #从后端开始截断或padding
    #因为maxLength太大了 会带来很大的计算消耗，因此这里将maxLength设置为300
    #maxLength=300
    X=pad_sequences(sequences,maxlen=maxLength,padding='post', truncating='post')

    return X,maxLength,patients_tokenizer

def drug_tokenizer(drugL,drugSet):
    drug2id={drug:id for id,drug in enumerate(drugSet)}
    drugIds=[]
    for line in drugL:
        drugIds.append([drug2id[item] for item in line])
    return drugIds,drug2id
def get_torch_sparse_matrix(A, dev):
    '''
    A : list of sparse adjacency matrices
    '''
    #idx:(2,2586)
    # print('A.tocoo().row:',A.tocoo().row)
    # print('A.tocoo().col:',A.tocoo().col)

    idx = torch.LongTensor([A.tocoo().row, A.tocoo().col])
    #dat :tensor，shape:(2586,)
    #print('A:',A.tocoo().data)
    dat = torch.FloatTensor(A.tocoo().data)
    return torch.sparse.FloatTensor(idx, dat, torch.Size([A.shape[0], A.shape[1]])).cuda(1)



