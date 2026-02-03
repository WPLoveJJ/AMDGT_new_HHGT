import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设 hypergt.py 在同一目录下
from hypergt import HyperGT 

class DualHyperGT(nn.Module):
    def __init__(self, args, 
                 n_drug_nodes, n_dis_nodes, 
                 n_drug_hes, n_dis_hes,
                 drug_in_channels, dis_in_channels,
                 hidden_channels, out_channels):
        super(DualHyperGT, self).__init__()
        
        # === 1. 药物分支 HyperGT ===
        self.drug_encoder = HyperGT(
            num_tokens=n_drug_nodes,       # 节点数
            num_nodes=n_drug_nodes,        # 节点数 (用于位置编码)
            in_channels=drug_in_channels,  # 输入特征维度
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # 编码输出维度
            num_hes=n_drug_hes,            # 药物超边总数 (即 H_drug_final.shape[1])
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            # 以下参数根据是否有邻接矩阵调整，默认仅使用 Transformer + H编码
            use_bn=True,
            use_residual=True,
            use_edge_loss=False,           # 如果不传入 G 矩阵做正则化，设为 False
            rb_order=0,                    # 关系偏置阶数，无 adjs 时设为 0
            use_jk=False
        )

        # === 2. 疾病分支 HyperGT ===
        self.dis_encoder = HyperGT(
            num_tokens=n_dis_nodes,
            num_nodes=n_dis_nodes,
            in_channels=dis_in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_hes=n_dis_hes,             # 疾病超边总数
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_bn=True,
            use_residual=True,
            use_edge_loss=False,
            rb_order=0,
            use_jk=False
        )

        # === 3. 融合预测模块 ===
        # 将药物和疾病特征拼接后进行预测
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_channels, 1)  # 输出预测分数
        )

    def forward(self, args, x_drug, H_drug, x_dis, H_dis):
        """
        x_drug: 药物特征 [N_drug, F_drug]
        H_drug: 药物关联矩阵 (Sparse Tensor) [N_drug, M_drug]
        """
        
        # HyperGT 的 forward 需要: args, x, adjs, H
        # 这里 adjs 传空列表 []，因为我们主要依赖 H 进行结构编码
        
        # 1. 提取药物特征
        # 输出形状: [N_drug, hidden_channels]
        drug_emb = self.drug_encoder(args, x_drug, [], H_drug)
        
        # 2. 提取疾病特征
        # 输出形状: [N_dis, hidden_channels]
        dis_emb = self.dis_encoder(args, x_dis, [], H_dis)
        
        return drug_emb, dis_emb

    def predict(self, drug_emb, dis_emb, drug_indices, dis_indices):
        """
        根据索引进行配对预测
        drug_indices, dis_indices: 形状相同的索引列表
        """
        batch_drug = drug_emb[drug_indices]
        batch_dis = dis_emb[dis_indices]
        
        cat_feat = torch.cat([batch_drug, batch_dis], dim=1)
        score = self.classifier(cat_feat)
        return torch.sigmoid(score)