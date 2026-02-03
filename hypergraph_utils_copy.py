import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

# 1. 计算欧氏距离矩阵
def Eu_dis(x):
    # x: (n_samples, n_features)
    dist_mat = cdist(x, x, 'euclid')
    return dist_mat

# 2. 计算皮尔逊相关系数矩阵
def Pear_corr(x):
    x = x.T
    x_pd = pd.DataFrame(x)
    dist_mat = x_pd.corr()
    return dist_mat.to_numpy()

# 3. 拼接多个超图关联矩阵 (支持多尺度 K)
def Multi_omics_hyperedge_concat(*H_list):
    H = None
    for h in H_list:
        if h is not None and h != []:
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H

# 4. 核心构建逻辑：KNN + 概率权重
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, edge_type='euclid'):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
  
    # === 修改点 1: 避免除以0或数值不稳定 ===
    # 这一步通常不需要改 dis_mat，但在计算权重时要注意

    if edge_type == 'euclid':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0  # 确保自己到自己的距离为0
            dis_vec = dis_mat[center_idx]  # 取出当前节点到所有其他节点的距离
            # 1. 排序找邻居 (从小到大)
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

            # === 修改点 2: 改进 sigma (avg_dis) 的计算 ===
            # 不要用全局平均，只用第 1 到 第 K 个邻居的平均距离 (排除自己，自己是0)
            # 这样可以保证 bandwidth 适应局部密度
            # 计算平均距离 (用于后续的高斯核缩放)
            local_neighbors_dist = dis_vec[nearest_idx[1:k_neig+1]]
            # avg_dis = np.average(dis_vec)
            avg_dis = np.mean(local_neighbors_dist)                 # <--- 修正：只计算局部K个邻居的平均距离

            # 防止 avg_dis 为 0 (比如所有邻居都是重复数据)
            if avg_dis == 0:
                avg_dis = 1e-6 # 给一个极小值，防止除0错误

            # 2. 强制包含自身 (Self-loop)
            # 如果前 K 个邻居里没把自己算进去 (通常 index 0 就是自己，但为了保险)，
            # 强行把第 K 个位置替换成自己。
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx
            # 3. 填充矩阵 H
            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    # === 核心公式：高斯核权重 ===
                    # 距离越小(dis_vec小)，指数部分越接近0，结果越接近1。
                    # avg_dis 用来做自适应缩放，防止不同数据的尺度影响太大。    
                    # H[node_idx, center_idx] = np.exp(-dis_vec[node_idx] ** 2 / (m_prob * avg_dis) ** 2)#高斯核函数

                    dst = dis_vec[node_idx]
                    # 如果是自身，距离为0，权重自然为1，保留。
                    # 如果是重复数据的邻居，距离为0，权重也为1，这在数学上是合理的（完全相似）。
                    # 但对于非重复邻居，现在权重会有区分度了。
                    
                    # === 核心公式优化 ===
                    # 只有当 m_prob 调整合适，且 avg_dis 是局部距离时，这里才会出现 0.5, 0.8 等中间值
                    w = np.exp(-dst ** 2 / (m_prob * avg_dis) ** 2)
                    H[node_idx, center_idx] = w
                else:
                    H[node_idx, center_idx] = 1.0  # 二值化：只要是邻居就是 1
                    
    elif edge_type == 'pearson':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = -999.
            # ^^^ 这是一个小 Trick，或者说可能的小 Bug/特殊处理。
            # 通常 Pearson 相关系数自己对自己是 1.0 (最大)。
            # 这里设为 -999 可能意图是想在 argsort 时把它排到最后？
            # 但下面的逻辑是找最大的值，所以这行代码有点反直觉，见下文分析。
            dis_vec = dis_mat[center_idx]
             # 1. 排序找邻居 (argsort 默认从小到大)
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
            # 反转 (变成从大到小)，因为相关系数越大越相似
            nearest_idx = nearest_idx[::-1] 
            
             # ... (强制包含自身的逻辑同上) ...
             # 3. 填充矩阵 H
            avg_dis = np.average(dis_vec) 
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    # === 核心公式：Pearson 权重 ===
                    # dis_vec 取值 [-1, 1]。
                    # 加 1.0 是为了保证指数部分为负，或者单纯为了平移。
                    # 这里的公式意图是：相关性越高，权重越大。  
                    H[node_idx, center_idx] = 1. - np.exp(-(dis_vec[node_idx]+1.0) ** 2 )
                else:
                    H[node_idx, center_idx] = 1.0
    return H

# 5. 主调用接口
def construct_hypergraph(features, K_neigs, is_probH, m_prob, edge_type='euclid'):
    """
    参数:
    features: 特征矩阵 (numpy array 或 tensor), shape (样本数, 特征数)
    K_neigs: 邻居数量列表, 例如 [10]
    is_probH: 是否构建带权重的概率超图
    edge_type: 'euclid' (欧氏距离) 或 'pearson' (皮尔逊相关系数)
    
    返回:
    H: 关联矩阵 (Tensor)
    """
    # 如果是 Tensor，转为 numpy
    if isinstance(features, torch.Tensor):
        X = features.cpu().detach().numpy()
    else:
        X = features

    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    # if isinstance(K_neigs, int):
    #     K_neigs = [K_neigs]

    # === 修改点 3: 添加微小噪声 (Jitter) ===
    # 这一步非常关键！用于解决 Mol2Vec 中完全重复的行。
    # 加一个极小的随机数 (1e-6 级别)，不会影响特征本质，但能让距离不再绝对为 0
    jitter = np.random.normal(0, 1e-6, X.shape)
    X_jittered = X + jitter

    # # 使用加噪后的数据计算距离
    if edge_type == 'euclid':
        # dis_mat = Eu_dis(X)
        dis_mat = Eu_dis(X_jittered)
    elif edge_type == 'pearson':
        dis_mat = Pear_corr(X)
        # dis_mat = Pear_corr(X_jittered) # Pearson 一般不需要 jitter，因为它是看趋势
    else:
        raise ValueError("edge_type must be 'euclid' or 'pearson'")

    # 构建矩阵
    H = None
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob, edge_type)
        H = Multi_omics_hyperedge_concat(H, H_tmp)
    
      # === 修改开始：转换为 PyTorch 稀疏张量 ===
    # 1. 先转为 dense tensor
    H_dense = torch.FloatTensor(H)
    
    # 2. 获取非零元素的索引
    # torch.nonzero 返回 shape (N, 2)，我们需要转置为 (2, N) 以符合 sparse_coo_tensor 格式
    indices = torch.nonzero(H_dense).t()
    
    # 3. 获取对应的数值
    values = H_dense[indices[0], indices[1]]
    
    # 4. 构建稀疏张量
    H_sparse = torch.sparse_coo_tensor(indices, values, H_dense.shape)
    
    return H_sparse.coalesce()
    # === 修改结束 ===