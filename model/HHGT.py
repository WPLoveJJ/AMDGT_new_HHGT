import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class HHGTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, node_types, dropout=0.2):
        super(HHGTLayer, self).__init__()
        self.heads = num_heads
        self.d_k = out_channels // num_heads
        self.dropout = dropout
        self.out_channels = out_channels
        
        # 定义异构投影矩阵
        self.k_lin = nn.ModuleDict({t: nn.Linear(in_channels, out_channels) for t in node_types})
        self.q_lin = nn.ModuleDict({t: nn.Linear(in_channels, out_channels) for t in node_types})
        self.v_lin = nn.ModuleDict({t: nn.Linear(in_channels, out_channels) for t in node_types})
        self.a_lin = nn.ModuleDict({t: nn.Linear(out_channels, out_channels) for t in node_types})
        self.skip = nn.ModuleDict({t: nn.Linear(in_channels, out_channels) for t in node_types})
        
        self.layernorm = nn.ModuleDict({t: nn.LayerNorm(out_channels) for t in node_types})

    def forward(self, x_dict, hyperedge_index_dict, max_he_id):
        # 1. 线性投影 & Reshape [N, Heads, D_k]
        H_node = {}
        for ntype, x in x_dict.items():
            H_node[ntype] = self.k_lin[ntype](x).view(-1, self.heads, self.d_k)

        # 2. Node -> Hyperedge Aggregation
        # 初始化超边特征存储器
        he_features = torch.zeros(max_he_id + 1, self.heads, self.d_k, device=list(x_dict.values())[0].device)
        
        for (ntype, _), indices in hyperedge_index_dict.items():
            # indices: [2, E] -> [0]: node_idx, [1]: he_idx
            node_idx, he_idx = indices[0], indices[1]
            x_source = H_node[ntype][node_idx]
            
            # 聚合：将节点特征加到对应的超边上 (Sum Pooling)
            he_features.index_add_(0, he_idx, x_source)

        # 3. Hyperedge -> Node Attention
        out_dict = {ntype: [] for ntype in x_dict}
        
        for (ntype, _), indices in hyperedge_index_dict.items():
            node_idx, he_idx = indices[0], indices[1]
            
            # Query 来自目标节点
            x_query = self.q_lin[ntype](x_dict[ntype]).view(-1, self.heads, self.d_k)[node_idx]
            # Key 来自所属超边
            x_key = he_features[he_idx]
            
            # Attention 计算
            alpha = (x_query * x_key).sum(dim=-1) / math.sqrt(self.d_k)
            # 简单的 Softmax 可能不够，通常配合 scatter_softmax，这里简化处理
            alpha = torch.softmax(alpha, dim=0) # 简化的 Attention
            
            val = x_key * alpha.unsqueeze(-1)
            
            # 聚合回节点
            node_out = torch.zeros(x_dict[ntype].size(0), self.heads, self.d_k, device=x_dict[ntype].device)
            node_out.index_add_(0, node_idx, val)
            out_dict[ntype].append(node_out)

        # 4. Merge & Residual
        res_dict = {}
        for ntype in x_dict:
            if out_dict[ntype]:
                # 合并多种超边来源的信息
                merged = sum(out_dict[ntype]).flatten(1)
                merged = self.a_lin[ntype](merged)
                # 残差 + LayerNorm
                res_dict[ntype] = self.layernorm[ntype](merged + self.skip[ntype](x_dict[ntype]))
            else:
                res_dict[ntype] = self.skip[ntype](x_dict[ntype])
                
        return res_dict

class HHGT(nn.Module):
    def __init__(self, in_dim, hidden_dim, heads, num_layers, node_types):
        super(HHGT, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HHGTLayer(in_dim, hidden_dim, heads, node_types))
            in_dim = hidden_dim # 下一层输入即为上一层输出

    def forward(self, x_dict, hyperedge_index_dict):
        # 计算最大的超边 ID，用于初始化 buffer
        max_id = 0
        for idx in hyperedge_index_dict.values():
            if idx.numel() > 0:
                max_id = max(max_id, idx[1].max().item())
        
        for layer in self.layers:
            x_dict = layer(x_dict, hyperedge_index_dict, max_id)
        return x_dict