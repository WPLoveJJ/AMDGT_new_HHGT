import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.2):
        super(HypergraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads

        # 定义 Q, K, V 的线性变换
        # 注意：这里我们让节点作为 Query，超边作为 Key 和 Value
        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)
        self.W_V = nn.Linear(in_dim, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU()

        # 如果输入维度和输出维度不一致，需要对残差进行投影
        # (虽然在修改后的 HypergraphTransformer 中通常不会触发，但为了代码健壮性保留)
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(self, x, H):
        """
        x: 节点特征矩阵 [num_nodes, in_dim]
        H: 超图关联矩阵 (Sparse Tensor) [num_nodes, num_hyperedges]
        """
        residual = x
        num_nodes = x.shape[0]

        # === 1. 生成节点 Query ===
        # Q_node: [num_nodes, num_heads, d_k]
        Q_node = self.W_Q(x).view(num_nodes, self.num_heads, self.d_k)

        # === 2. 生成超边特征 (Node -> Hyperedge Aggregation) ===
        # 我们需要先得到“超边”的特征表示，作为 Key 和 Value
        # 公式：E = H^T * X (简单的聚合，也可以归一化)
        
        # 为了计算方便，我们先转换 H 为 dense (如果显存允许，药物/疾病数量少完全没问题)
        # 或者使用 sparse.mm
        if H.is_sparse:
             # H_T: [num_hyperedges, num_nodes]
            H_T = H.t()
            # Hyperedge features: [num_hyperedges, in_dim]
            x_hyperedge = torch.sparse.mm(H_T, x)
            
            # 归一化：除以超边的度 (包含多少个节点)
            deg_edge = torch.sparse.sum(H, dim=0).to_dense().clamp(min=1).unsqueeze(1)
            x_hyperedge = x_hyperedge / deg_edge
        else:
            x_hyperedge = torch.matmul(H.t(), x)

        # === 3. 生成超边 Key 和 Value ===
        # K_edge, V_edge: [num_hyperedges, num_heads, d_k]
        num_hyperedges = x_hyperedge.shape[0]
        K_edge = self.W_K(x_hyperedge).view(num_hyperedges, self.num_heads, self.d_k)
        V_edge = self.W_V(x_hyperedge).view(num_hyperedges, self.num_heads, self.d_k)

        # === 4. 计算注意力 (Hyperedge -> Node Attention) ===
        # 节点关注它所属的（或相似的）超边
        # Attention Score = (Q_node * K_edge^T) / sqrt(d_k)
        # 维度: [num_nodes, num_heads, num_hyperedges]
        
        # 这里的矩阵乘法：(N, H, D) * (M, H, D)^T -> (N, H, M)
        scores = torch.einsum('nhd, mhd -> nhm', Q_node, K_edge) / (self.d_k ** 0.5)
        
        # 这里的 mask 可选：强制节点只关注它所在的超边
        # 如果 H 是 0/1 矩阵，可以用 H 作为 mask；如果是概率矩阵，直接用即可
        # 这里我们做全注意力（借鉴Transformer全局视野），让节点可以关注到语义相似的超边
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # === 5. 聚合信息 ===
        # Out = Attn * V_edge
        # 维度: [num_nodes, num_heads, M] * [M, num_heads, D] -> [num_nodes, num_heads, D]
        out = torch.einsum('nhm, mhd -> nhd', attn, V_edge)
        
        # === 6. 拼接多头并输出 ===
        out = out.contiguous().view(num_nodes, self.out_dim)
        out = self.out_proj(out)

         # === 6. 残差连接处理 ===
        # 如果维度不匹配，先对 residual 做投影
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # 残差连接 + 归一化
        out = self.layer_norm(residual + out)
        
        return out

class HypergraphTransformer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, num_heads, dropout):
        super(HypergraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        
        # 第一层：输入映射到隐藏层
        self.layers.append(HypergraphAttentionLayer(in_dim, hidden_dim, num_heads, dropout))
        
        # 后续层
        for _ in range(num_layers - 1):
            self.layers.append(HypergraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout))
            
        # 最后的输出映射（如果 hidden != out）
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, H):
        for layer in self.layers:
            x = layer(x, H)
        return self.final_proj(x)