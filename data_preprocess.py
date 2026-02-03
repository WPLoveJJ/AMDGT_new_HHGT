import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import os

device = torch.device('cuda')

#将边列表转换为邻接矩阵
def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
        #print(knn_graph[i, idx_sort[i, :k + 1]])#除去自己，前21个最强邻居节点。
        """i为0 ->[0.57793558 0.52793558 0.47542555 0.43501238 0.43501238 0.42793558
 0.42793558 0.42793558 0.39616345 0.38501237 0.37997145 0.37997145
 0.37542555 0.37542555 0.37542555 0.36001237 0.33724426 0.33501238
 0.33501238 0.33501238 0.33501238]"""
        #print(knn_graph[idx_sort[i, :k + 1], i])
        """[0.57793558 0.52793558 0.47542555 0.43501238 0.43501238 0.42793558
 0.42793558 0.42793558 0.39616345 0.38501237 0.37997145 0.37997145
 0.37542555 0.37542555 0.37542555 0.36001237 0.33724426 0.33501238
 0.33501238 0.33501238 0.33501238]"""
    return knn_graph + np.eye(num)


def get_data(args):
    data = dict()
    #Drug Fingerprint (药物分子指纹)
    drf = pd.read_csv(args.data_dir + 'DrugFingerprint.csv').iloc[:, 1:].to_numpy()
    #Drug GIP (药物的高斯交互概型)
    drg = pd.read_csv(args.data_dir + 'DrugGIP.csv').iloc[:, 1:].to_numpy()#读取药物GIP相似矩阵
    #Disease PS / Phenotype (疾病表型相似性)
    dip = pd.read_csv(args.data_dir + 'DiseasePS.csv').iloc[:, 1:].to_numpy()
    #Disease GIP (疾病的高斯交互概型)
    dig = pd.read_csv(args.data_dir + 'DiseaseGIP.csv').iloc[:, 1:].to_numpy()#读取疾病GIP相似矩阵
    # print (drg)
    # print(dig)
    data['drug_number'] = int(drf.shape[0])
    data['disease_number'] = int(dig.shape[0])
    # print(data['drug_number'])
    # print(data['disease_number'])
    # exit(0)
    data['drf'] = drf
    data['drg'] = drg
    data['dip'] = dip
    data['dig'] = dig
    #DDrug-Disease Interactions (药物-疾病关联)
    data['drdi'] = pd.read_csv(args.data_dir + 'DrugDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    #Drug-Protein Interactions (药物-靶标蛋白关联)
    data['drpr'] = pd.read_csv(args.data_dir + 'DrugProteinAssociationNumber.csv', dtype=int).to_numpy()
    #Disease-Protein Interactions (疾病-蛋白关联)
    data['dipr'] = pd.read_csv(args.data_dir + 'ProteinDiseaseAssociationNumber.csv', dtype=int).to_numpy()
    #初始特征向量
    data['drugfeature'] = pd.read_csv(args.data_dir + 'Drug_mol2vec.csv', header=None).iloc[:, 1:].to_numpy()
    data['diseasefeature'] = pd.read_csv(args.data_dir + 'DiseaseFeature.csv', header=None).iloc[:, 1:].to_numpy()
    data['proteinfeature'] = pd.read_csv(args.data_dir + 'Protein_ESM.csv', header=None).iloc[:, 1:].to_numpy()
    data['protein_number']= data['proteinfeature'].shape[0]#统计蛋白质的总个数

    return data


def data_processing(data, args):
    drdi_matrix = get_adj(data['drdi'], (args.drug_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(drdi_matrix.shape[0]):
        for j in range(drdi_matrix.shape[1]):
            if drdi_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    unsamples = zero_index[int(args.negative_rate * len(one_index)):]
    data['unsample'] = np.array(unsamples)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    drdi_p = samples[samples[:, 2] == 1, :]
    drdi_n = samples[samples[:, 2] == 0, :]
    # 对应 公式 5 和 公式 6 的平均值计算
    drs_mean = (data['drf'] + data['drg']) / 2
    dis_mean = (data['dip'] + data['dig']) / 2
    # 对应 Eq. 5 和 Eq. 6 的条件判断与融合 (如果某项为0则用另一项或均值填充)
    drs = np.where(data['drf'] == 0, data['drg'], drs_mean)
    dis = np.where(data['dip'] == 0, data['dip'], dis_mean)

    data['drs'] = drs 
    data['dis'] = dis
    data['all_samples'] = samples
    data['all_drdi'] = samples[:, :2]
    data['all_drdi_p'] = drdi_p
    data['all_drdi_n'] = drdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p

    return data


def k_fold(data, args):
    k = args.k_fold

    for i in range(k):
        # 动态创建目录
        output_dir = os.path.join(args.data_dir, 'fold', str(i))
        os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=k, random_state=None, shuffle=False)
    X = data['all_drdi']
    Y = data['all_label']
    # n = skf.get_n_splits(X, Y)
    X_train_all, X_test_all, Y_train_all, Y_test_all = [], [], [], []
    for train_index, test_index in skf.split(X, Y):
        # print('Train:', train_index, 'Test:', test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_train = np.expand_dims(Y_train, axis=1).astype('float64')
        Y_test = np.expand_dims(Y_test, axis=1).astype('float64')
        X_train_all.append(X_train)
        X_test_all.append(X_test)
        Y_train_all.append(Y_train)
        Y_test_all.append(Y_test)

    for i in range(k):
        X_train1 = pd.DataFrame(data=np.concatenate((X_train_all[i], Y_train_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_train1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_train.csv')
        X_test1 = pd.DataFrame(data=np.concatenate((X_test_all[i], Y_test_all[i]), axis=1), columns=['drug', 'disease', 'label'])
        X_test1.to_csv(args.data_dir + 'fold/' + str(i) + '/data_test.csv')

    data['X_train'] = X_train_all
    data['X_test'] = X_test_all
    data['Y_train'] = Y_train_all
    data['Y_test'] = Y_test_all
    return data


def dgl_similarity_graph(data, args):
    drdr_matrix = k_matrix(data['drs'], args.neighbor)#data['drs']融合后的药物相似性矩阵DRs(ri,rj),稠密矩阵；k_matrix只保留相似度最高的 args.neighbor 个邻居（例如前 20 个）。
    didi_matrix = k_matrix(data['dis'], args.neighbor)#data['dis']融合后的疾病相似性矩阵DSs(si,sj),稠密矩阵
    drdr_nx = nx.from_numpy_matrix(drdr_matrix)#构建图的拓扑结构 (NetworkX)-将上一步得到的 NumPy 邻接矩阵转化为图对象-NetworkX 运行在 CPU 上
    didi_nx = nx.from_numpy_matrix(didi_matrix)
    #drdr_nx 是一个包含节点和边的 Python 对象
    #drdr_nx = nx.from_numpy_array(drdr_matrix)
    #didi_nx = nx.from_numpy_array(didi_matrix)
    drdr_graph = dgl.from_networkx(drdr_nx)#转换为 DGL 图格式 (DGL Conversion),运行在GPU上
    didi_graph = dgl.from_networkx(didi_nx)
    #print(data['drs'])
    #print(drdr_matrix[0])
    """[[1. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 [0. 0. 0. ... 0. 0. 1.]]
   
[[1.         0.         0.40786213 ... 0.         0.         0.        ]
 [0.         1.         0.15225705 ... 0.         0.         0.        ]
 [0.40786213 0.15225705 1.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 1.         0.36770015 0.        ]
 [0.         0.         0.         ... 0.36770015 1.         0.        ]
 [0.         0.         0.         ... 0.         0.         1.        ]]"""
    #使用drs和dis作为节点特征，即它的初始特征是它与所有其他药物的相似度向量，第i节点的特征为第i行相似度向量
    drdr_graph.ndata['drs'] = torch.tensor(data['drs'])
    didi_graph.ndata['dis'] = torch.tensor(data['dis'])
   # print(drdr_graph.ndata['drs'][1])

    return drdr_graph, didi_graph, data


def dgl_heterograph(data, drdi, args):
    drdi_list, drpr_list, dipr_list = [], [], []
    for i in range(drdi.shape[0]):
        drdi_list.append(drdi[i])
    for i in range(data['drpr'].shape[0]):
        drpr_list.append(data['drpr'][i])
    for i in range(data['dipr'].shape[0]):
        dipr_list.append(data['dipr'][i])

    node_dict = {
        'drug': args.drug_number,
        'disease': args.disease_number,
        'protein': args.protein_number
    }

    heterograph_dict = {
        ('drug', 'association', 'disease'): (drdi_list),
        ('drug', 'association', 'protein'): (drpr_list),
        ('disease', 'association', 'protein'): (dipr_list)
    }

    data['feature_dict'] ={
        'drug': torch.tensor(data['drugfeature']),
        'disease': torch.tensor(data['diseasefeature']),
        'protein': torch.tensor(data['proteinfeature'])
    }

    drdipr_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)

    return drdipr_graph, data





