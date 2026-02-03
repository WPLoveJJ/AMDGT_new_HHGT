"""
SAITS model for time-series imputation.

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


from modeling.layers import *
from modeling.utils import masked_mae_cal


class SAITS(nn.Module):
    def __init__(
        self,
        n_groups,  # 组的数量
        n_group_inner_layers,  # 每个组内的层数
        d_feature,  # 特征的维度
        d_model,  # 模型的维度
        d_inner,  # 内部层的维度
        n_head,  # 多头注意力机制中的头数
        d_k,  # 键的维度
        d_v,  # 值的维度
        dropout,  # Dropout层的概率
        **kwargs  # 额外的关键字参数
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]  # 决定了输入是否包含掩码
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature  # 如果输入包含掩码，则特征维度翻倍；否则，使用原始特征维度
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]  # 获取参数共享策略
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":  # 检查参数共享策略
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_group_inner_layers)
                ]
            )
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )
            self.layer_stack_for_second_block = nn.ModuleList(
                [
                    EncoderLayer(
                        d_model,
                        d_model,
                        d_inner,
                        n_head,
                        d_k,
                        d_v,
                        dropout,
                        **kwargs
                    )
                    for _ in range(n_groups)
                ]
            )

        self.dropout = nn.Dropout(p=dropout)  # 创建一个Dropout层，用于防止过拟合
        # for the 1st block
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)  # 创建第一个线性层，用于将输入特征映射到模型维度
        self.reduce_dim_z = nn.Linear(d_model, d_feature)  # 创建一个线性层，用于将模型维度映射回特征维度
        # for the 2nd block
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for the 3rd block
        self.weight_combine = nn.Linear(d_feature * 2, d_feature)  # 创建一个线性层，用于计算加权组合的权重

    # 执行数据插补
    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]  # 从输入中提取特征数据和掩码

        #if X.dim() == 2:
         #   X = X.unsqueeze(1)  # [663, 400] → [663, 1, 400]
        # if masks.dim() == 2:
         #   masks = masks.unsqueeze(1)  # [663, 400] → [663, 1, 400]

        # the first DMSA block
        input_X_for_first = torch.cat([X, masks], dim=-1) if self.input_with_mask else X  # 如果输入包含掩码，则将特征和掩码拼接；否则，只使用特征
        #print(f"input_X_for_first shape: {input_X_for_first.shape}")
        input_X_for_first = self.embedding_1(input_X_for_first)  # 将拼接后的数据通过第一个线性层进行映射
        #print(f"After embedding_1: {input_X_for_first.shape}")
        enc_output = self.dropout(input_X_for_first)  # namely term e in math algo 应用Dropout层
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)  # 对每个组重复应用第一个块中的编码层
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        X_tilde_1 = self.reduce_dim_z(enc_output)  # 将编码输出通过线性层映射回特征维度，得到第一次插补的结果
        X_prime = masks * X + (1 - masks) * X_tilde_1  # 根据掩码将原始特征和插补特征进行组合

        # the second DMSA block
        input_X_for_second = (
            torch.cat([X_prime, masks], dim=-1) if self.input_with_mask else X_prime
        )  # 如果输入包含掩码，则将插补后的特征和掩码拼接；否则，只使用插补后的特征
        input_X_for_second = self.embedding_2(input_X_for_second)  # 将拼接后的数据通过第二个线性层进行映射
        enc_output = input_X_for_second  # namely term alpha in math algo
        if self.param_sharing_strategy == "between_group":
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)

        # 将编码输出通过线性层和ReLU激活函数映射回特征维度，得到第二次插补的结果
        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # the attention-weighted combination block
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in math algo 将注意力权重的维度压缩，去掉单一维度
        if len(attn_weights.shape) == 4:
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)

            # 关键：将attn_weights从[663, 1, 1]扩展到[663, 1, d_feature]
        attn_weights = attn_weights.expand(-1, -1, X.size(-1))  # 扩展特征维度，与masks匹配

        # 确保masks与attn_weights形状一致（防止意外维度不匹配）
        if masks.shape != attn_weights.shape:
            masks = masks.expand_as(attn_weights)


        combining_weights = F.sigmoid(
            self.weight_combine(torch.cat([masks, attn_weights], dim=-1))
        )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # replace non-missing part with original data
        X_c = masks * X + (1 - masks) * X_tilde_3
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]  # 返回最终的插补数据和中间结果

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        reconstruction_loss = 0  # 初始化重建损失为0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)  # 调用 impute 函数执行插补，并获取插补数据和中间结果

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)  # 计算第一次插补结果的重建损失
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)  # 计算最终重建损失
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3  # 将总重建损失平均

        if (self.MIT or stage == "val") and stage != "test":  # 检查是否需要计算插补损失
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            # 计算插补损失
            imputation_MAE = masked_mae_cal(
                X_tilde_3, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)  # 否则，将插补损失设为0

        # 返回插补数据、重建损失、插补损失和其他指标
        return {
            "imputed_data": imputed_data,
            "X_tilde_1": X_tilde_1,  # 第一个子网络输出
            "X_tilde_2": X_tilde_2,  # 第二个子网络输出
            "X_tilde_3": X_tilde_3,  # 第三个子网络输出
            "reconstruction_loss": reconstruction_loss,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": final_reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }