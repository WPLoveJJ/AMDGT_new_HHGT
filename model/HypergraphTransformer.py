import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. 辅助函数：生成随机投影矩阵 (用于线性 Attention) ===
def create_projection_matrix(m, d, seed=0, scaling=0, struct_mode=False):
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed
    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        unstructured_block = torch.randn((d, d))
        # q, _ = torch.qr(unstructured_block)
        q, _ = torch.linalg.qr(unstructured_block, mode='reduced') 
        q = torch.t(q)
        block_list.append(q)
        current_seed += 1
    
    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        unstructured_block = torch.randn((d, d))
        # q, _ = torch.qr(unstructured_block)
        q, _ = torch.linalg.qr(unstructured_block, mode='reduced')
        q = torch.t(q)
        block_list.append(q[0:remaining_rows])
    
    final_matrix = torch.vstack(block_list)
    current_seed += 1
    torch.manual_seed(current_seed)
    
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}")

    return torch.matmul(torch.diag(multiplier), final_matrix)

def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = ratio * torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    
    # 这里的Exp对应Softmax Kernel
    return torch.exp(data_dash) + numerical_stabilizer

# === 2. 稀疏线性层 (用于处理超图关联矩阵 H) ===
class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # 如果输入是稀疏矩阵，使用 sparse.mm
        if input.is_sparse:
            wb = torch.sparse.mm(input, self.weight.T)
        else:
            wb = torch.mm(input, self.weight.T)
            
        if self.bias is not None:
            return wb + self.bias
        return wb

# === 3. 核心卷积层 (NodeFormer Attention) ===
class HyperGTConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, nb_random_features=30):
        super(HyperGTConv, self).__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.nb_random_features = nb_random_features

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()

    def forward(self, x):
        # x: [Batch=1, N, Dim] 或者是 [N, Dim]
        if x.dim() == 2:
            x = x.unsqueeze(0) # [1, N, C]
            
        B, N, C = x.shape
        
        # Linear Projections
        Q = self.Wq(x).view(B, N, self.num_heads, self.out_channels)
        K = self.Wk(x).view(B, N, self.num_heads, self.out_channels)
        V = self.Wv(x).view(B, N, self.num_heads, self.out_channels)
        
        # 生成随机投影矩阵
        projection_matrix = create_projection_matrix(self.nb_random_features, self.out_channels).to(x.device)
        
        # Kernel Transformation
        Q_prime = softmax_kernel_transformation(Q, True, projection_matrix)
        K_prime = softmax_kernel_transformation(K, False, projection_matrix)
        
        # Linear Attention: (Q' @ (K'.T @ V))
        # K_prime: [B, N, H, M], V: [B, N, H, D] -> KV: [B, H, M, D]
        KV = torch.einsum("bnhm,bnhd->bhmd", K_prime, V)
        
        # Z: [B, N, H, D]
        Z_num = torch.einsum("bnhm,bhmd->bnhd", Q_prime, KV)
        
        # Normalization denominator
        K_prime_sum = torch.sum(K_prime, dim=1) # [B, H, M]
        Z_den = torch.einsum("bnhm,bhm->bnh", Q_prime, K_prime_sum).unsqueeze(-1)
        
        Z = Z_num / (Z_den + 1e-6)
        
        # Concat heads and Output projection
        Z = Z.reshape(B, N, -1)
        output = self.Wo(Z)
        
        return output.squeeze(0) # [N, Out_Dim]

# === 4. 主模型类 ===
class HypergraphTransformer(nn.Module):
    def __init__(self, device, num_layers, num_nodes, num_hyperedges, in_dim, out_dim, num_heads, dropout=0.0):
        super(HypergraphTransformer, self).__init__()
        self.device = device
        self.num_layers = num_layers#2
        self.dropout = dropout
        
        # 1. 维度变换
        self.input_fc = nn.Linear(in_dim, out_dim)
        
        # 2. 超边位置编码 (Hyperedge Positional Encoding)
        # 利用关联矩阵 H 对节点进行位置编码
        self.he_sparse_encoder = SparseLinear(num_hyperedges, out_dim)#SparseLinear特殊线性层，针对稀疏输入进行优化，降低计算开销并提高内存效率；num_hyperedges=1989，超边的总数量，out_dim(200),输出的维度，嵌入的尺寸，输出200维度的稠密矩阵
    
        # 3. Transformer 层堆叠
        self.convs = nn.ModuleList()#创建专门存放卷积层的容器
        self.bns = nn.ModuleList()#创建专门存放归一化层的容器
        
        for _ in range(num_layers):#循环构建多层结构
            self.convs.append(
                HyperGTConv(out_dim, out_dim, num_heads=num_heads)
            )#定义超图卷积/转换层，每一层都会通过注意力机制更新节点和超边的表示，捕捉数据中的空间拓扑结构
            self.bns.append(nn.LayerNorm(out_dim))#添加归一化层，防止在深层网络中梯度消失或爆炸，确保每一层输出的均值和方差稳定
            
        self.activation = F.elu#设置激活函数ELU，与 ReLU 不同，ELU 在 x < 0 时有负值输出，这使得神经元的平均激活值更接近于零，从而加快学习速度，它在左侧是平滑的，比 ReLU 更抗噪

    def forward(self, x, H):
        """
        x: 节点特征 [N, in_dim]
        H: 超图关联矩阵 [N, E] (可以是稀疏张量或稠密张量)
        """
        # 1. 初始特征投影
        x = self.input_fc(x)
        
        # 2. 注入超图结构信息 (Positional Encoding)
        # 这是 HyperGT 的核心：利用 H 矩阵学习结构编码
        if H.device != self.device:
            H = H.to(self.device)
            
        # 确保 H 是 Float 类型
        if not H.is_sparse and H.dtype != torch.float32:
            H = H.float()
            
        he_pe = self.he_sparse_encoder(H) # [N, out_dim]
        
        # 将结构编码加到特征上
        x = x + he_pe
        
        # 3. 层层传递
        for i in range(self.num_layers):
            x_in = x
            x = self.bns[i](x) # Pre-Norm
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x)
            x = x + x_in # Residual connection
            
        return x