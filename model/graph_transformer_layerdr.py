import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
    Graph Transformer Layer
    
"""
"""
    Util functions
"""
# 辅助函数：计算源节点和目标节点特征之间的点积，并将结果存储在指定字段中。
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        # 计算 Attention 中的分子部分：Query * Key
        # 对应论文公式 (9) 中的 Q_{ij}^k * K_{ij}^k
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

# 辅助函数：对边上的注意力得分进行缩放并取指数，以确保数值稳定性（防止过大或过小值）。
def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        # 对分数除以 sqrt(d_k) 进行缩放，并限制数值范围防止溢出，最后取指数
        # 对应论文公式 (9) 中的 softmax 操作前的 exp(Score / sqrt(d))
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
        #edges.data[field]: $QK^T$
        #scale_constant 对应 $\sqrt{d_k}$
        #$e^{x_i}$
        #torch.exp 对应 Softmax 内部的指数操作
        #.clamp(-5, 5)将缩放后的分数强制限制在 [-5, 5] 的范围内
    return func


"""
    Single Attention Head
"""
class MultiHeadAttentionLayer(nn.Module):#多头注意力层
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()#初始化
        
        self.out_dim = out_dim  # 每个头的输出维度
        self.num_heads = num_heads  # 注意力头的数量

        # 初始化Q、K、V变换矩阵，用于生成查询、键和值向量  
        # 对应论文公式 (8):
        # Q_{ij}^k = Q^k * h_i^0
        # K_{ij}^k = K^k * h_j^0
        # V_{ij}^k = V^k * h_j^0
        #假设输入特征是 512 维，你要用 8 个头，每个头算出 64 维的特征
        #nn.Linear(..., bias=True)：计算 y = xW^T + b。引入一个可学习的偏差，增加模型的拟合能力。
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
    
    def propagate_attention(self, g):
        # 计算注意力得分（点积相似度）
        # 使用上面定义的 src_dot_dst 计算 K_h * Q_h
        # 对应论文公式 (9) 分子部分
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) 
        # 对注意力得分进行缩放和指数运算，为后续softmax做准备
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        ## 自定义消息函数替代 src_mul_edge
        # 将 Value (V_h) 与 计算出的注意力权重 (score) 相乘
        def message_func(edges):
            return {'V_h': edges.src['V_h'] * edges.data['score']}

        # 将加权的值发送到目标节点，并聚合信息
        eids = g.edges()
        # V_h * score -> wV: 加权后的值聚合到目标节点
        # 对应论文公式 (10) 中的 Sum(A * V)
        #g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, message_func, fn.sum('V_h', 'wV'))#将各个节点的特征Vi*注意力权重score_ij，Σv_h=wV中
        # score -> z: 注意力得分之和，用于归一化
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))#fn.sum('score', 'z')->Σei
    

    def forward(self, g, h ,hg):

        # 应用线性变换生成Q、K、V
        Q_h = self.Q(h)# 计算 Query
        K_h = self.K(h)# 计算 Key
        V_h = self.V(h)# 计算 Value

        # 计算 H_T 
        H_T = hg.t()
        # 稀疏矩阵乘法
        HH_T = torch.sparse.mm(hg, H_T)

        Q_h = Q_h.view(-1, self.out_dim * self.num_heads)
        # 稀疏矩阵与密集矩阵乘法
        Q_h = torch.sparse.mm(HH_T, Q_h)

        Q_h = Q_h.view(-1, self.num_heads, self.out_dim)

        # 调整形状 [num_nodes, num_heads, feat_dim] 以便于多头注意力机制操作
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)


        self.propagate_attention(g)
        # 归一化加权后的值
        head_out = g.ndata['wV']/g.ndata['z']#Σ（e_ij*V）/Σ（e_ij） 公式10括号的结果
        return head_out
    

class GraphTransformerLayer(nn.Module):
    """
        Param: 
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            num_heads: 多头注意力机制中的头数
            dropout: Dropout率
            layer_norm: 是否使用层归一化
            batch_norm: 是否使用批归一化
            residual: 是否使用残差连接
            use_bias: 是否在线性层中使用偏置
    """
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super().__init__()

        self.in_channels = in_dim  # 输入通道数（特征维度）
        self.out_channels = out_dim  # 输出通道数（特征维度）
        self.num_heads = num_heads  # 注意力头数量
        self.dropout = dropout  # Dropout率
        self.residual = residual  # 是否启用残差连接
        self.layer_norm = layer_norm  # 是否启用层归一化
        self.batch_norm = batch_norm  # 是否启用批归一化
   
        # 初始化多头注意力机制层
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        #print( self.attention)
        #print("in_dim="+str(in_dim),"out_dim//num_heads="+ str(out_dim//num_heads),"num_heads="+str(num_heads), "use_bias="+str(use_bias))
        
        # 线性变换层O，用于将多头注意力输出映射回原始维度
        self.O = nn.Linear(out_dim, out_dim)

        # 如果启用了层归一化，则初始化层归一化层
        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)
            
        # 如果启用了批归一化，则初始化批归一化层
        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)
        
        # 前馈神经网络（FFN），包括两个线性变换层
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        # 如果启用了层归一化，则初始化第二个层归一化层
        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)
            
        # 如果启用了批归一化，则初始化第二个批归一化层
        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)


    def forward(self, g, h ,hg):
        h_in1 = h  # 保存输入用于第一个残差连接
       # print(h)
        # 多头注意力机制输出  # 1. 获取多头输出 (这里是 [N, num_heads, out_dim//num_heads])
        attn_out = self.attention(g, h, hg)
        # 2. 拼接操作 ||
        # self.out_channels 等于 总维度 (out_dim)
        # view 操作将 (头数, 单头维度) 两个维度展平合并为一个维度 8*64 = 512
        h = attn_out.view(-1, self.out_channels)  # 调整形状

        # 应用Dropout
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 通过线性变换层O 
        # 对应标准 Transformer 中的 Output Matrix W_O
        # 作用是把拼接后的特征融合一下，不仅仅是简单的拼起来
        h = self.O(h)
        
        # 如果启用了残差连接，则添加原始输入与经过变换后的输入
        if self.residual:
            h = h_in1 + h  # 第一个残差连接 
           #print(h)
        
        # 如果启用了层归一化，则应用层归一化
        if self.layer_norm:
            h = self.layer_norm1(h)
            
        # 如果启用了批归一化，则应用批归一化
        if self.batch_norm:
            h = self.batch_norm1(h)
        
        h_in2 = h  # 保存当前状态用于第二个残差连接
        
        # 前馈神经网络（FFN）处理
        h = self.FFN_layer1(h)
        h = F.relu(h)  # 应用ReLU激活函数
        h = F.dropout(h, self.dropout, training=self.training)  # 再次应用Dropout
        h = self.FFN_layer2(h)

        # 如果启用了残差连接，则再次添加原始输入与经过变换后的输入
        if self.residual:
            h = h_in2 + h  # 第二个残差连接
        
        # 如果启用了层归一化，则应用第二个层归一化
        if self.layer_norm:
            h = self.layer_norm2(h)
            
        # 如果启用了批归一化，则应用第二个批归一化
        if self.batch_norm:
            h = self.batch_norm2(h)       

        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)