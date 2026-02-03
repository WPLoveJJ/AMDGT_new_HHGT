class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args 

        print(f"SAITS d_feature: {args.d_feature}, AMDGT out dim: {args.gt_out_dim}")
        
        # 维度检查
        assert args.d_feature == 2 * args.gt_out_dim, "特征维度必须匹配！"
        
        # 1. 输入投影层 (用于 HHGT 输入)
        # 将药物原始特征 (300) -> 64
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        # 将蛋白质原始特征 (320) -> 64
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
       
        # 2. 超图 Transformer (用于另一路特征)
        # 这里的 out_dim = 200
        self.gt_drug = HypergraphTransformer(
            device=device,
            num_layers=args.gt_layer,
            num_nodes=args.drug_number,
            num_hyperedges=args.drug_number * 3, 
            in_dim=300,               # 药物原始维度
            out_dim=args.gt_out_dim,  # 输出 200
            num_heads=args.gt_head,
            dropout=args.amdgt_dropout
        )

        self.gt_disease = HypergraphTransformer(
            device=device,
            num_layers=args.gt_layer,
            num_nodes=args.disease_number,
            num_hyperedges=args.disease_number * 3,
            in_dim=64,                # 疾病原始维度
            out_dim=args.gt_out_dim,  # 输出 200
            num_heads=args.gt_head,
            dropout=args.amdgt_dropout
        )

        # 3. HHGT 模型
        node_types = ['drug', 'disease', 'protein']
        self.hhgt = HHGT(
            in_dim=args.hgt_in_dim,       # 输入 64
            hidden_dim=args.hgt_in_dim,   # 隐藏层 64 -> 输出也是 64
            heads=args.hgt_head,
            num_layers=args.hgt_layer,
            node_types=node_types
        )
        
        # === 【关键修复】新增：HHGT 输出投影层 ===
        # 将 HHGT 的输出 (64维) 映射到 GT 的输出维度 (200维) 以便 stack
        self.hhgt_out_proj = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        # ======================================

        # 4. 后续 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        
        # 5. MLP 分类器
        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )
          
        # 6. SAITS 模型
        self.saits_drug = SAITS(
            n_groups=args.n_groups,
            n_group_inner_layers=args.n_group_inner_layers,
            d_feature=args.d_feature,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.saits_dropout,
            input_with_mask=args.input_with_mask,
            param_sharing_strategy=args.param_sharing_strategy,
            MIT=args.MIT,
            device=device
        )
        
        self.saits_disease = SAITS(
            n_groups=args.n_groups,
            n_group_inner_layers=args.n_group_inner_layers,
            d_feature=args.d_feature,
            d_model=args.d_model,
            d_inner=args.d_inner,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.saits_dropout,
            input_with_mask=args.input_with_mask,
            param_sharing_strategy=args.param_sharing_strategy,
            MIT=args.MIT,
            device=device
        )

    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample,
                H_drdr, H_didi):
        
        H_drdr = H_drdr.to(device)
        H_didi = H_didi.to(device)
        
        # === 1. Hypergraph Transformer 分支 (输出 200 维) ===
        dr_sim = self.gt_drug(drug_feature, H_drdr)      # [Batch, 200]
        di_sim = self.gt_disease(disease_feature, H_didi)# [Batch, 200]
        
        # === 2. HHGT 分支 (输出 64 维 -> 投影到 200 维) ===
        # (A) 准备输入 (300/320 -> 64)
        drug_feat_hgt = self.drug_linear(drug_feature)
        protein_feat_hgt = self.protein_linear(protein_feature)
        disease_feat_hgt = disease_feature # 假设原本就是 64

        x_dict = {
            'drug': drug_feat_hgt,
            'disease': disease_feat_hgt,
            'protein': protein_feat_hgt
        }
        
        # (B) 运行 HHGT
        hyperedge_index_dict = dgl_hetero_to_hyperedge_index(drdipr_graph, device)
        out_dict = self.hhgt(x_dict, hyperedge_index_dict)
        
        dr_hgt_raw = out_dict['drug']   # [Batch, 64]
        di_hgt_raw = out_dict['disease'] # [Batch, 64]
        
        # (C) 【关键修复】投影对齐维度 (64 -> 200)
        dr_hgt = self.hhgt_out_proj(dr_hgt_raw) # [Batch, 200]
        di_hgt = self.hhgt_out_proj(di_hgt_raw) # [Batch, 200]

        # === 3. 特征堆叠与融合 ===
        # 现在两个都是 [Batch, 200]，可以 stack 了
        dr = torch.stack((dr_sim, dr_hgt), dim=1) # [Batch, 2, 200]
        di = torch.stack((di_sim, di_hgt), dim=1) # [Batch, 2, 200]
        
        dr = self.drug_trans(dr)
        di = self.disease_trans(di)
        
        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim) # [Batch, 400]
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        # === 4. SAITS 部分 (保持不变) ===
        M_drug_original = (dr != 0).float().to(device) 
        I_drug = torch.rand_like(dr) < 0.2 
        I_drug = I_drug.float() * M_drug_original 
        M_hat_drug = M_drug_original - I_drug

        M_di_original = (di != 0).float().to(device)
        I_di = torch.rand_like(di) < 0.2
        I_di = I_di.float() * M_di_original
        M_hat_di = M_di_original - I_di

        dr_3d = dr.unsqueeze(1)
        dr_saits_input = dr_3d * M_hat_drug.unsqueeze(1)
        drug_inputs = {"X": dr_saits_input, "missing_mask": M_hat_drug.unsqueeze(1)}

        di_3d = di.unsqueeze(1)
        di_saits_input = di_3d * M_hat_di.unsqueeze(1)
        disease_inputs = {"X": di_saits_input, "missing_mask": M_hat_di.unsqueeze(1)}

        drug_saits_out = self.saits_drug.impute(drug_inputs)
        dr_saits = drug_saits_out[0]
        X_tilde1_dr, X_tilde2_dr, X_tilde3_dr = drug_saits_out[1]

        disease_saits_out = self.saits_disease.impute(disease_inputs)
        di_saits = disease_saits_out[0]
        X_tilde1_di, X_tilde2_di, X_tilde3_di = disease_saits_out[1]

        dr_saits = dr_saits.squeeze(1)
        di_saits = di_saits.squeeze(1)

        mit_loss_drug = masked_mae_cal(dr_saits, dr, I_drug)
        ort_loss_drug = (masked_mae_cal(X_tilde1_dr.squeeze(1), dr, M_hat_drug) +
                         masked_mae_cal(X_tilde2_dr.squeeze(1), dr, M_hat_drug) +
                         masked_mae_cal(X_tilde3_dr.squeeze(1), dr, M_hat_drug)) / 3

        mit_loss_di = masked_mae_cal(di_saits, di, I_di)
        ort_loss_di = (masked_mae_cal(X_tilde1_di.squeeze(1), di, M_hat_di) +
                       masked_mae_cal(X_tilde2_di.squeeze(1), di, M_hat_di) +
                       masked_mae_cal(X_tilde3_di.squeeze(1), di, M_hat_di)) / 3

        drug_total_loss = mit_loss_drug + ort_loss_drug
        disease_total_loss = mit_loss_di + ort_loss_di 

        drdi_embedding = torch.mul(dr_saits[sample[:, 0]], di_saits[sample[:, 1]])
        output = self.mlp(drdi_embedding)

        return drug_total_loss, disease_total_loss, output