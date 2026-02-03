import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
import math

class HeteroHypergraphTransformerConv(nn.Module):
    def __init__(self, in_channels_dict, out_channels, metadata, heads=4, dropout=0.2):
        super().__init__()
        self.out_channels = out_channels
        self.heads = heads
        self.d_k = out_channels // heads
        self.dropout = dropout
        self.metadata = metadata # (node_types, edge_types)
        
        # 1. 定义异构投影矩阵 (与 HGT 相同)
        # 每个节点类型都有自己独立的 Q, K, V 变换矩阵
        self.k_lin = nn.ModuleDict()
        self.q_lin = nn.ModuleDict()
        self.v_lin = nn.ModuleDict()
        self.a_lin = nn.ModuleDict() # 用于最后的聚合
        
        for node_type, in_dim in in_channels_dict.items():
            self.k_lin[node_type] = nn.Linear(in_dim, out_channels)
            self.q_lin[node_type] = nn.Linear(in_dim, out_channels)
            self.v_lin[node_type] = nn.Linear(in_dim, out_channels)
            self.a_lin[node_type] = nn.Linear(out_channels, out_channels)

        # 2. 超边 Attention 参数
        # 我们需要两个 Attention：Node->Hyperedge 和 Hyperedge->Node
        self.attn_node_to_he = nn.Parameter(torch.ones(heads, self.d_k, self.d_k))
        self.attn_he_to_node = nn.Parameter(torch.ones(heads, self.d_k, self.d_k))
        
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.k_lin.values(): nn.init.xavier_uniform_(lin.weight)
        for lin in self.q_lin.values(): nn.init.xavier_uniform_(lin.weight)
        for lin in self.v_lin.values(): nn.init.xavier_uniform_(lin.weight)
        nn.init.xavier_uniform_(self.attn_node_to_he)
        nn.init.xavier_uniform_(self.attn_he_to_node)

    def forward(self, x_dict, hyperedge_index_dict):
        """
        x_dict: {node_type: x} 节点特征
        hyperedge_index_dict: {(node_type, he_type): [2, num_entries]} 
                              即 (node_idx, hyperedge_idx)
        """
        
        # -------------------------------------------------------
        # 第一步：准备特征 (Prepare Features)
        # -------------------------------------------------------
        H_node = {} # 变换后的节点特征
        for ntype, x in x_dict.items():
            # [N, Heads, D_k]
            H_node[ntype] = self.k_lin[ntype](x).view(-1, self.heads, self.d_k)

        # -------------------------------------------------------
        # 第二步：Node -> Hyperedge 聚合 (Node to Hyperedge Aggregation)
        # 这一步计算超边的表征。超边特征 = 包含的节点特征的加权和
        # -------------------------------------------------------
        hyperedge_features = {} # 存储临时超边特征
        
        for (ntype, he_type), edge_index in hyperedge_index_dict.items():
            node_idx, he_idx = edge_index
            
            # 获取参与该超边的节点特征
            x_source = H_node[ntype][node_idx] # [E, Heads, D_k]
            
            # 这里简化处理：直接使用 Mean Aggregation 或者简单的 Attention
            # 在复杂版本中，这里应该计算 Attention(Hyperedge_Query, Node_Key)
            # 这里我们假设超边也是一个"虚拟节点"，但初始化为空，所以先直接聚合节点信息
            
            # 使用 scatter_add 将节点特征聚合到超边上
            # 实际代码需要处理不同类型的节点聚合到同一个超边ID上的情况
            # 这里假设 he_idx 是全局统一的，或者我们按 he_type 处理
            
            # 简单实现：Sum Pooling
            out = torch.zeros(he_idx.max()+1, self.heads, self.d_k, device=x_source.device)
            out = out.scatter_add(0, he_idx.unsqueeze(-1).unsqueeze(-1).expand_as(x_source), x_source)
            
            if he_type not in hyperedge_features:
                hyperedge_features[he_type] = out
            else:
                hyperedge_features[he_type] += out # 不同类型的节点聚合到同一类超边

        # -------------------------------------------------------
        # 第三步：Hyperedge -> Node 更新 (Hyperedge to Node Update)
        # 节点通过 Query 去查询它所属的超边 (Key/Value)
        # -------------------------------------------------------
        out_node_features = {ntype: [] for ntype in x_dict}
        
        for (ntype, he_type), edge_index in hyperedge_index_dict.items():
            node_idx, he_idx = edge_index
            
            # Target (Node) 发出 Query
            # x_target: [E, Heads, D_k]
            x_query = self.q_lin[ntype](x_dict[ntype]).view(-1, self.heads, self.d_k)[node_idx]
            
            # Source (Hyperedge) 提供 Key 和 Value
            # hyperedge_features[he_type] 已经是聚合了节点信息的特征
            he_feat = hyperedge_features[he_type][he_idx] # [E, Heads, D_k]
            
            # 计算 Attention Score: (Query * W * Key^T)
            # 这里简化为 Dot Product Attention
            alpha = (x_query * he_feat).sum(dim=-1) / math.sqrt(self.d_k) # [E, Heads]
            
            # Softmax 归一化 (对每个节点归一化它的所有超边)
            alpha = softmax(alpha, node_idx) 
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            # 加权求和得到节点更新量
            out = he_feat * alpha.unsqueeze(-1) # [E, Heads, D_k]
            
            # 聚合回节点
            node_out = torch.zeros(x_dict[ntype].size(0), self.heads, self.d_k, device=x_dict[ntype].device)
            node_out = node_out.scatter_add(0, node_idx.unsqueeze(-1).unsqueeze(-1).expand_as(out), out)
            
            out_node_features[ntype].append(node_out)

        # -------------------------------------------------------
        # 第四步：合并与输出 (Merge and Output)
        # -------------------------------------------------------
        final_results = {}
        for ntype in x_dict:
            if len(out_node_features[ntype]) > 0:
                # 把来自不同类型超边的信息加起来
                summed = torch.stack(out_node_features[ntype]).sum(dim=0) # [N, Heads, D_k]
                # Flatten
                flattened = summed.view(-1, self.out_channels)
                # Residual + Norm + Activation (参考 HGT)
                res = flattened + x_dict[ntype] # 残差连接需要维度匹配，这里略去线性层
                final_results[ntype] = self.a_lin[ntype](res)
            else:
                final_results[ntype] = x_dict[ntype] # 孤立节点保持不变
                
        return final_results