"""
Layer modules for self-attention models (Transformer and SAITS).

If you use code in this repository, please cite our paper as below. Many thanks.

@article{DU2023SAITS,
title = {{SAITS: Self-Attention-based Imputation for Time Series}},
journal = {Expert Systems with Applications},
volume = {219},
pages = {119619},
year = {2023},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2023.119619},
url = {https://www.sciencedirect.com/science/article/pii/S0957417423001203},
author = {Wenjie Du and David Cote and Yan Liu},
}

or

Wenjie Du, David Cote, and Yan Liu. SAITS: Self-Attention-based Imputation for Time Series. Expert Systems with Applications, 219:119619, 2023. https://doi.org/10.1016/j.eswa.2023.119619
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: MIT


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """scaled dot-product attention
       实现了缩放点积注意力机制
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  # 缩放因子，用于控制注意力分数的范围
        self.dropout = nn.Dropout(attn_dropout)  # 注意力分数的丢弃率

    def forward(self, q, k, v, attn_mask=None):  # 查询张量、键张量、值张量、注意力掩码（用于屏蔽某些位置的注意力）
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # 计算缩放后的点积注意力分数,对应公式
        if attn_mask is not None:  # 将掩码位置的注意力分数设置为负无穷
            attn = attn.masked_fill(attn_mask == 1, -1e9)  # 负无穷经过 Softmax 后会趋近于 0，即这些位置的注意力权重被 “清零”，模型不会关注它们
        attn = self.dropout(F.softmax(attn, dim=-1))  # 对注意力分数沿最后一维做 Softmax 操作，将分数转换为和为 1 的概率分布，并施加 Dropout 防止过拟合
        output = torch.matmul(attn, v)  # 得到最终的注意力输出，形状为 (batch_size, num_heads, seq_len_q, d_v)
        return output, attn  # 返回注意力输出和注意力权重


class MultiHeadAttention(nn.Module):
    """original Transformer multi-head attention
       多头注意力
    """

    def __init__(self, n_head, d_model, d_k, d_v, attn_dropout):  # 并行注意力头的数量、输入特征的维度、每个头的键和值维度、丢失率
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        #  线性投影层：将输入特征投影到多个头的查询、键、值空间
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)


        # 缩放点积注意力层
        self.attention = ScaledDotProductAttention(d_k**0.5, attn_dropout)  # ** 0.5 是数学中的 “开平方” 运算
        # 输出线性层：将多头结果拼接后投影回原始维度
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

    #  前向传播函数
    def forward(self, q, k, v, attn_mask=None):
        #  获取参数和输入维度
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)  # 批量大小、查询/键/值的序列长度

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # 线性投影与多头分割


        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # 通过 self.w_qs线性层，将输入的 q投影到查询空间，使用 view 函数将投影后的张量重塑形状
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # 调整维度顺序
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 处理注意力掩码
        if attn_mask is not None:
            # this mask is imputation mask, which is not generated from each batch, so needs broadcasting on batch dim
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)  # For batch and head axis broadcasting.
            # 增加两个头维度，拓展后和注意力分数的形状对齐

        v, attn_weights = self.attention(q, k, v, attn_mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # 恢复维度顺序并拼接多头结果
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # contiguous() 确保张量在内存中连续存储,view() 将头数和头维度合并
        v = v.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        v = self.fc(v)  # 将拼接后的高维特征投影回原始维度
        return v, attn_weights  # 返回最终的多头注意力输出和注意力权重矩阵


# 位置前馈网络
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1): # 输入特征维度、隐藏层维度
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # 第一层线性变换：输入维度->隐藏维度
        self.w_2 = nn.Linear(d_hid, d_in)  # 第二层线性变换：隐藏维度->输出维度（输入维度）
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)  # 层归一化，对单个样本的所有特征进行归一化，稳定训练
        self.dropout = nn.Dropout(dropout)

    # 前向传播过程
    def forward(self, x):
        residual = x  # 保存输入作为残差连接
        x = self.layer_norm(x)  # 先进行层归一化
        x = self.w_2(F.relu(self.w_1(x)))  # 两层全连接：线性变换→ReLU（激活函数引入非线性）→线性变换
        x = self.dropout(x)  # 应用Dropout，防止过拟合
        x += residual  # 残差连接：输出 = 变换后特征 + 原始输入，缓解梯度消失问题
        return x


# 编码层
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_feature,  # 特征维度
        d_model,  # 模型维度
        d_inner,  # 前馈网络中间层维度
        n_head,  # 注意力头数
        d_k,  # 键的维度
        d_v,  # 值的维度
        dropout=0.1,  # Dropout 概率
        attn_dropout=0.1,  # 注意力权重的 Dropout 概率
        **kwargs  # 灵活传递参数
    ):
        super(EncoderLayer, self).__init__()

        self.d_feature = d_feature

        # 添加特征投影层（新增）
        self.feature_proj = nn.Linear(d_feature, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, attn_dropout)
        self.dropout = nn.Dropout(dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout)

    def forward(self, x):
        # 投影原始特征到模型维度
        x = self.feature_proj(x)

        residual = x  # 保存输入作为残差
        # here we apply LN before attention cal, namely Pre-LN, refer paper https://arxiv.org/abs/2002.04745
        x = self.layer_norm(x)
        # 多头自注意力计算，查询（Q）、键（K）、值（V）都来自同一输入，因此称为 “自注意力”
        enc_output, attn_weights = self.slf_attn(x, x, x, attn_mask=None)  # 无掩码，非时序数据允许全连接注意力
        enc_output = self.dropout(enc_output)
        enc_output += residual

        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_weights  # 返回编码层的最终输出和注意力权重
