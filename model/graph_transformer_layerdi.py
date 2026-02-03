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
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

# 辅助函数：对边上的注意力得分进行缩放并取指数，以确保数值稳定性（防止过大或过小值）。
def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func


"""
    Single Attention Head
"""
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        
        self.out_dim = out_dim  # 每个头的输出维度
        self.num_heads = num_heads  # 注意力头的数量

        # 初始化Q、K、V变换矩阵，用于生成查询、键和值向量
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
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) 
        # 对注意力得分进行缩放和指数运算，为后续softmax做准备
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        ## 自定义消息函数替代 src_mul_edge
        def message_func(edges):
            return {'V_h': edges.src['V_h'] * edges.data['score']}

        # 将加权的值发送到目标节点，并聚合信息
        eids = g.edges()
        # V_h * score -> wV: 加权后的值聚合到目标节点
        #g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, message_func, fn.sum('V_h', 'wV'))
        # score -> z: 注意力得分之和，用于归一化
        g.send_and_recv(eids, fn.copy_e('score', 'score'), fn.sum('score', 'z'))
    

    def forward(self, g, h, hg):

        # 应用线性变换生成Q、K、V
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

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
        head_out = g.ndata['wV']/g.ndata['z']

        
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
       
        # 多头注意力机制输出
        attn_out = self.attention(g, h, hg)
        h = attn_out.view(-1, self.out_channels)  # 调整形状

        # 应用Dropout
        h = F.dropout(h, self.dropout, training=self.training)
        
        # 通过线性变换层O
        h = self.O(h)
        
        # 如果启用了残差连接，则添加原始输入与经过变换后的输入
        if self.residual:
            h = h_in1 + h  # 第一个残差连接
        
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