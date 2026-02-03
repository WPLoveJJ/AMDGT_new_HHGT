"""
Transformer model for time-series imputation.

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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_groups,
        n_group_inner_layers,
        d_feature,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout,
        **kwargs
    ):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs["input_with_mask"]  # 决定了输入是否包含掩码
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature  # 如果输入包含掩码，则特征维度翻倍；否则，使用原始特征维度
        self.param_sharing_strategy = kwargs["param_sharing_strategy"]
        self.MIT = kwargs["MIT"]
        self.device = kwargs["device"]

        if kwargs["param_sharing_strategy"] == "between_group":
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        actual_d_feature,
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
            self.layer_stack = nn.ModuleList(
                [
                    EncoderLayer(
                        actual_d_feature,
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

        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim = nn.Linear(d_model, d_feature)

    def impute(self, inputs):
        X, masks = inputs["X"], inputs["missing_mask"]
        #if X.dim() == 2:
         #   X = X.unsqueeze(1)  # 从 [663, 400] → [663, 1, 400]
        #if masks.dim() == 2:
         #   masks = masks.unsqueeze(1)
        input_X = torch.cat([X, masks], dim=-1) if self.input_with_mask else X  # 如果输入包含掩码，则将特征和掩码拼接；否则，只使用特征
        input_X = self.embedding(input_X)  # 将拼接后的数据通过线性层映射到模型维度
        enc_output = self.dropout(input_X)  # 应用Dropout层

        if self.param_sharing_strategy == "between_group":
            # 对每个组重复应用 layer_stack 中的编码层
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            # 对每个编码层重复应用指定次数
            for encoder_layer in self.layer_stack:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)

        learned_presentation = self.reduce_dim(enc_output)  # 将编码输出通过线性层映射回原始特征维度
        imputed_data = (
            masks * X + (1 - masks) * learned_presentation  # 根据掩码将原始特征和插补特征进行组合
        )  # replace non-missing part with original data
        return imputed_data, learned_presentation  # 返回插补数据和学习到的表示

    def forward(self, inputs, stage):
        X, masks = inputs["X"], inputs["missing_mask"]
        imputed_data, learned_presentation = self.impute(inputs)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)  # 计算重建的平均绝对误差（MAE）
        if (self.MIT or stage == "val") and stage != "test":
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(
                learned_presentation, inputs["X_holdout"], inputs["indicating_mask"]
            )
        else:
            imputation_MAE = torch.tensor(0.0)

        return {
            "imputed_data": imputed_data,
            "reconstruction_loss": reconstruction_MAE,
            "imputation_loss": imputation_MAE,
            "reconstruction_MAE": reconstruction_MAE,
            "imputation_MAE": imputation_MAE,
        }
