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
    #  # 1. 过滤掉 None 和空列表
    valid_H = [h for h in H_list if h is not None and h != []]
    
    if not valid_H:
        return None

    # 2. 检查第一个元素的类型，决定使用 PyTorch 还是 Numpy
    first_H = valid_H[0]

    # === 情况 A: 输入是 PyTorch Tensor (包括 Sparse Tensor) ===
    if isinstance(first_H, torch.Tensor):
        # 使用 torch.cat 代替 np.hstack
        # dim=1 表示沿着列（超边）拼接
        # 注意：PyTorch 稀疏张量支持 torch.cat
        return torch.cat(valid_H, dim=1)
    
    # === 情况 B: 输入是 Numpy Array ===
    else:
        H = None
        for h in valid_H:
            if H is None:
                H = h
            else:
                # 原有的 Numpy 拼接逻辑
                if isinstance(h, list):
                     # 某些特殊情况如果 h 是 list，递归处理或特殊处理
                     # 这里为了健壮性，假设 h 也是 numpy
                     pass 
                H = np.hstack((H, h))
    return H


# 4. 核心构建逻辑：KNN + 概率权重
def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1, edge_type='euclid'):
    n_obj = dis_mat.shape[0]
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
  
    if edge_type == 'euclid':
        for center_idx in range(n_obj):
            dis_mat[center_idx, center_idx] = 0  # 确保自己到自己的距离为0
            dis_vec = dis_mat[center_idx]        # 取出当前节点到所有其他节点的距离
            
            # 1. 排序找邻居 (从小到大)
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()

            # 2. 自适应 Sigma 计算 (Local Sigma)
            # 只计算局部K个邻居的平均距离 (排除自己，从索引1开始)
            # 如果 K=1，取第二个节点（索引1）的距离
            if k_neig < n_obj - 1:
                local_neighbors_dist = dis_vec[nearest_idx[1:k_neig+1]]
            else:
                local_neighbors_dist = dis_vec[nearest_idx[1:]]

            avg_dis = np.mean(local_neighbors_dist)
            
            # 防止 avg_dis 为 0 (比如所有邻居都是完全重复数据)
            if avg_dis < 1e-9:
                avg_dis = 1e-9 

            # 3. 强制包含自身 (Self-loop)
            # 逻辑：如果前K个里没有自己，把第K个位置(索引 k_neig-1)替换成自己
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx

            # 4. 填充矩阵 H
            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    dst = dis_vec[node_idx]
                    # 核心公式：高斯核权重
                    w = np.exp(-dst ** 2 / (m_prob * avg_dis) ** 2)
                    H[node_idx, center_idx] = w
                else:
                    H[node_idx, center_idx] = 1.0
                    
    elif edge_type == 'pearson':
        for center_idx in range(n_obj):
            # === 修正: 自身相关性设为 1.0 (最大) ===
            # 这样排序后，自身自然会排在第一个，逻辑更顺
            dis_mat[center_idx, center_idx] = 1.0
            dis_vec = dis_mat[center_idx]
            
            # 1. 排序找邻居 (降序，从大到小)
            # argsort 默认升序，[::-1] 翻转为降序
            nearest_idx = np.array(np.argsort(dis_vec)).squeeze()[::-1]

            # 2. 强制包含自身 (保险起见，虽然上面设了1.0通常不需要这步)
            if not np.any(nearest_idx[:k_neig] == center_idx):
                nearest_idx[k_neig - 1] = center_idx
            # 3. 计算自适应/局部 Sigma (可选，为了与 Euclid 逻辑高度一致)
            # 取前 K 个邻居的相关系数，转换为“距离”后计算平均值
            if is_probH:
                # 映射：corr=1 -> dist=0; corr=0 -> dist=1; corr=-1 -> dist=2
                local_dists = 1.0 - dis_vec[nearest_idx[:k_neig]]
                avg_dist = np.mean(local_dists)
                if avg_dist < 1e-9: avg_dist = 1e-9
            # 4. 填充矩阵 H
            for node_idx in nearest_idx[:k_neig]:
                if is_probH:
                    # Pearson 值域 [-1, 1]
                    # 越接近 1，权重越大
                    # 使用简单的映射，或者也可以引入 m_prob 进行控制
                    # 原公式: 1 - exp(-(corr+1)^2) -> corr=1时 w~1; corr=-1时 w=0
                    val = dis_vec[node_idx]
                    # 核心公式转换
                    dist_pearson = 1.0 - val
                    w = np.exp(-dist_pearson**2 / (m_prob * avg_dist)**2)
                    H[node_idx, center_idx] = w
                else:
                    H[node_idx, center_idx] = 1.0
                    
    return H

# 5. 主调用接口
def construct_hypergraph(features, K_neigs, is_probH=True, m_prob=1.0, edge_type='euclid'):
    """
    参数:
    features: 特征矩阵 (numpy array 或 tensor), shape (样本数, 特征数)
    K_neigs: 邻居数量列表, 例如 [10] (支持 int 或 list)
    is_probH: 是否构建带权重的概率超图
    m_prob: 权重缩放因子
    edge_type: 'euclid' 或 'pearson'
    
    返回:
    H: 关联矩阵 (PyTorch Sparse Tensor)
    """
    # 如果是 Tensor，转为 numpy，将输入数据统一转换为 NumPy 数组
    if isinstance(features, torch.Tensor):
        X = features.cpu().detach().numpy()
    else:
        X = features.copy() # 避免修改原数据

    if len(X.shape) != 2:#将数据展平为二维矩阵
        X = X.reshape(-1, X.shape[-1])

    # === 修正: 恢复类型检查，防止 int 报错 ===
    if isinstance(K_neigs, int):
        K_neigs = [K_neigs]

    # === 关键: 添加微小噪声 (Jitter) ===
    # 解决完全重复数据导致距离为0、无法区分邻居的问题
    # 1e-6 足够小，不改变特征含义，但能打破排序平局
    #向特征中添加极小的正态分布噪声
    jitter = np.random.normal(0, 1e-6, X.shape)
    X_jittered = X + jitter

    # 计算距离矩阵
    if edge_type == 'euclid':#欧氏距离：计算空间中点与点的直线距离
        # 使用加噪数据计算距离
        dis_mat = Eu_dis(X_jittered)
    elif edge_type == 'pearson':#皮尔逊相关系数：计算特征间的相关性
        # Pearson 通常不需要 Jitter，因为它看的是趋势
        # 但如果向量完全一样，corr 可能会出 NaN，所以加上 Jitter 更安全
        dis_mat = Pear_corr(X_jittered)
        # 处理可能的 NaN (如果向量方差为0)
        dis_mat = np.nan_to_num(dis_mat, nan=0.0) #如果某个特征向量的所有值都相同（方差为 0），相关系数计算会出现 0/0导致 NaN，这里将其转为 0
    else:
        raise ValueError("edge_type must be 'euclid' or 'pearson'")

    # 构建矩阵
    H = None
    for k_neig in K_neigs:
        # 确保 K 不超过样本数
        k_real = min(k_neig, X.shape[0])
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_real, is_probH, m_prob, edge_type)#根据距离矩阵，为每个节点找到最近的 K个邻居，并形成一条超边
        #行代表节点，列代表超边。如果节点 $i$ 属于超边 $j$，则 H_{i,j} > 0
        H = Multi_omics_hyperedge_concat(H, H_tmp)#将不同 K值生成的超边水平拼接在一起
    
    # === 转换为 PyTorch 稀疏张量 ===
    H_dense = torch.FloatTensor(H)
    
    # 获取非零元素的索引并转置
    indices = torch.nonzero(H_dense).t()
    #返回形式如 [[row1, col1], [row2, col2], ...]
    #.t() 表示转置操作
    # 获取对应的数值
    values = H_dense[indices[0], indices[1]]#利用刚才得到的索引，从原稠密矩阵中把对应的非零数值（如概率权重或 1）提取出来。
    
    # 构建稀疏张量 (Size: N_nodes x Total_Hyperedges)
    H_sparse = torch.sparse_coo_tensor(indices, values, H_dense.shape)#indices: 坐标信息，values: 数值信息，H_dense.shape: 矩阵的原始维度（行数=节点数，列数=总超边数）
    #COO（Coordinate）格式稀疏矩阵的标准索引表示法：第一行是行索引，第二行是列索引
    return H_sparse.coalesce()#coalesce() 意为“合并”


def generate_G_from_H(H, variable_weight=False):
    """ 
       根据关联矩阵 H 计算超图 G 矩阵
        G = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    """
    # === 兼容性处理: 如果是 PyTorch Tensor，先转回 Dense Numpy ===
    if isinstance(H, torch.Tensor):
        if H.is_sparse:
            H = H.to_dense().cpu().numpy()
        else:
            H = H.cpu().numpy()
            
    # === 核心计算 (使用 Numpy 以保证数值稳定) ===
    H = np.array(H)
    n_edge = H.shape[1]
    
    # 超边权重 W，这里默认全为 1
    W = np.ones(n_edge)
    
    # 节点的度 DV (行和)
    DV = np.sum(H * W, axis=1)
    # 超边的度 DE (列和)
    DE = np.sum(H, axis=0)

    # 避免除以 0，加微小量 1e-5
    invDE = np.mat(np.diag(np.power(DE + 1e-5, -1)))
    DV2 = np.mat(np.diag(np.power(DV + 1e-5, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        # 核心公式
        G = DV2 * H * W * invDE * HT * DV2
        # 计算完成后转回 Tensor
        return torch.Tensor(G).float()

def dgl_hetero_to_hyperedge_index(g, device):
    """
    将 DGL 异构图转换为 HHGT 需要的 hyperedge_index_dict
    逻辑：将图中的每一条边视为一个超边 (size=2)
    """
    hyperedge_index_dict = {}
    he_id_counter = 0
    
    # 遍历 DGL 图中所有的边类型 (canonical_etypes)
    # etype 格式通常是 ('source_type', 'relation', 'dest_type')
    for etype in g.canonical_etypes:
        src_type, rel, dst_type = etype
        
        # 获取源节点和目标节点的索引 (Tensor)
        src_idx, dst_idx = g.edges(etype=etype)
        num_edges = src_idx.shape[0]
        
        if num_edges == 0: continue

        src_idx = src_idx.to(device)
        dst_idx = dst_idx.to(device)
        
        # 分配超边 ID
        he_ids = torch.arange(he_id_counter, he_id_counter + num_edges, device=device)
        
        # 1. Source -> Hyperedge
        key_src = (src_type, rel)
        # 堆叠为 [2, Num_Edges]
        hyperedge_index_dict[key_src] = torch.stack([src_idx, he_ids], dim=0)
        
        # 2. Dest -> Hyperedge
        key_dst = (dst_type, rel)
        # 如果是自环或者同类节点，key需要区分吗？通常 HHGT 内部按 key 遍历即可
        # 这里的 key 只要唯一标识 "某种节点连接到某类超边" 即可
        # 为了防止 key 冲突 (比如 drug->treats 和 disease->treated_by)，我们可以加后缀
        if key_dst in hyperedge_index_dict:
             # 如果冲突（极为罕见，除非定义的 key 不够唯一），需要特殊处理
             pass
        hyperedge_index_dict[key_dst] = torch.stack([dst_idx, he_ids], dim=0)
        
        he_id_counter += num_edges
        
    return hyperedge_index_dict        