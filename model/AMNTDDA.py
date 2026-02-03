import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease
from modeling.saits import SAITS
from modeling.transformer import TransformerEncoder
from modeling.utils import masked_mae_cal
from torch.utils.data import Dataset, DataLoader
from modeling.loss_functions import mit_loss, ort_loss
from model.HypergraphTransformer import HypergraphTransformer
from model.HHGT import HHGT 
from hypergraph_utils import dgl_hetero_to_hyperedge_index # 导入刚才新写的函数

device = torch.device('cuda')


class AMNTDDA(nn.Module):
    def __init__(self, args):

        super(AMNTDDA, self).__init__()# 调用父类初始化-torch.nn.Module，这是 PyTorch 标准写法
        self.args = args # 将传入的超参数字典保存到类中，方便全局使用

        print(f"SAITS d_feature: {args.d_feature}, AMDGT out dim: {args.gt_out_dim}")
        # 这是一个关键的断言检查！
        # 这里的逻辑是：后续会将 "GraphTransformer特征" 和 "HGT特征" 拼接。
        # 如果两者维度都是 gt_out_dim，拼接后就是 2 * gt_out_dim。
        # 而 SAITS 模型需要的输入维度 (d_feature) 必须等于这个拼接后的维度
        assert args.d_feature == 2 * args.gt_out_dim, "特征维度必须匹配！"  #assert -Python 的断言关键字，判断条件，false就抛出"特征维度必须匹配！"异常
        #  输入投影层 (用于 HHGT 输入)
        # 将药物原始特征（300维）映射到 HGT 需要的输入维度，将不同维度的原始特征，映射到同一个目标维度空间中
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        # 将蛋白质原始特征（320维）映射到 HGT 需要的输入维度，hgt_in_dim为变换后的目标维度
        # 将蛋白质原始特征 (320) -> 64
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
       
        # === 修改部分开始：使用 Hypergraph Transformer ===
        # 假设 args 中有 drug_number (N) 和 hyperedge_number (E)
        # H_drdr 的形状通常是 [drug_number, drug_number] (如果作为 KNN 构建)，
        # 或者 [drug_number, num_hyperedges]。此处假设 KNN 构图，E=N。
        
        # 初始化药物超图 Transformer
        self.gt_drug = HypergraphTransformer(
            device=device,
            num_layers=args.gt_layer,
            num_nodes=args.drug_number,
            num_hyperedges=args.drug_number * 3, # 假设 KNN 构图，超边数通常等于节点数
            # in_dim=args.gt_out_dim,          # 注意：这里需要确认输入的特征维度
            in_dim=300,
            out_dim=args.gt_out_dim,# 输出 200
            num_heads=args.gt_head,
            dropout=args.amdgt_dropout
        )

        # 初始化疾病超图 Transformer
        self.gt_disease = HypergraphTransformer(
            device=device,
            num_layers=args.gt_layer,
            num_nodes=args.disease_number,
            num_hyperedges=args.disease_number * 3,
            in_dim=64,              # 疾病原始维度
            out_dim=args.gt_out_dim,# 输出 200
            num_heads=args.gt_head,
            dropout=args.amdgt_dropout
        )
        # === 修改部分结束 ===

        #初始化异质图神经网络
        # self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head), args.hgt_head,
        #                                            3, 3, args.amdgt_dropout)
        #[1] 输入特征维度 (承接上一层的输出)[2] 单个注意力头的维度 (head_size) [3] 注意力头的数量 [4] 节点类型数量 (num_ntypes)3种不同节点，药物、疾病、蛋白质 [5] 边类型数量 (num_etypes)3种不同边 [6] Dropout 比率
        #输出投影层
        # self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3,
        #                                                 args.amdgt_dropout)
        # === 2. 修改：替换原本的 DGL HGT 定义 ===
        # 3. HHGT 模型
        # 定义节点类型列表 (根据你的 drdipr_graph)
        node_types = ['drug', 'disease', 'protein']
        
        self.hhgt = HHGT(
            in_dim=args.hgt_in_dim,       #  # 输入 64维度 (经过 linear 变换后的)
            hidden_dim=args.hgt_in_dim,   # 隐藏层 64 -> 输出也是 64
            heads=args.hgt_head,          # 头数
            num_layers=args.hgt_layer,    # 层数
            node_types=node_types         # 节点类型
        )
        # === 修改结束 === 


        # # --- 真正构建 HGT 层列表 --- 空的容器，专门用来存放神经网络层
        # self.hgt = nn.ModuleList()
        #print("args.hgt_layer="+str(args.hgt_layer))
        # #args.hgt_layer=2
        # 循环创建前 (Layer - 1) 层 (隐藏层)
        # for l in range(args.hgt_layer - 1):
        #     self.hgt.append(
        #         dgl.nn.pytorch.conv.HGTConv(
        #             args.hgt_in_dim,
        #             int(args.hgt_in_dim / args.hgt_head),# Head Size (每个头的维度)，强行让输出总维度等于输入维度，保持维度不变
        #             args.hgt_head,
        #             3, 3,
        #             args.amdgt_dropout  # 使用独立的 AMDGT dropout 参数
        #         )
        #     )
        #     #append 是 Python 列表（List），将dgl.nn.pytorch.conv.HGTConv放到列表里去，它的作用是**「堆叠」**神经网络层。
        # self.hgt.append(
        #     dgl.nn.pytorch.conv.HGTConv(
        #         args.hgt_in_dim,# [1] 输入特征维度
        #         args.hgt_head_dim,# [2] 单个注意力头的维度 (重点在这里！)
        #         args.hgt_head,# [3] 注意力头的数量
        #         3, 3,# [4] 节点类型数量 (num_ntypes)# [5] 边类型数量 (num_etypes)
        #         args.amdgt_dropout  # 使用独立的 AMDGT dropout 参数
        #     )
        # )

         # === 【关键修复】新增：HHGT 输出投影层 ===
        # 将 HHGT 的输出 (64维) 映射到 GT 的输出维度 (200维) 以便 stack
        self.hhgt_out_proj = nn.Linear(args.hgt_in_dim, args.gt_out_dim)
        # ======================================

        # 1. 定义单层模板,用于特征提取
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        #2. 堆叠多层
        #构建药物的特征提取塔，使用上面定义的“砖头”（encoder_layer），堆叠 num_layers 层，构建出完整的药物特征编码器
        #专门用于处理药物特征序列，通过自注意力机制增强特征表示
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)#[1]特征维度 [2]多头注意力机制的“头数”
        #构建疾病的特征提取塔，同一个 encoder_layer 变量作为模板，但 drug_trans 和 disease_trans 是两个独立的对象，它们的权重参数（Weights）是独立训练的，互不干扰。
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        
        # #构建药物的完整 Transformer
        # self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3,
        #                               num_decoder_layers=3, batch_first=True)#num_decoder_layers=3：解码器也固定为 3 层
        #                               #true：输入数据的形状必须是 (批次大小, 序列长度, 特征维度)。
        # #构建疾病的完整 Transformer
        # self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3,
        #                                  num_decoder_layers=3, batch_first=True)
        #定义MLP（多层感知器）用于最终分类
        self.mlp = nn.Sequential(  #nn.Sequential 是一个“管道”容器
            nn.Linear(args.gt_out_dim * 2, 1024),#特征融合与升维/保持,融合后的特征映射到1024维空间，
            nn.ReLU(),#激活函数
            nn.Dropout(0.4),#丢弃层，在训练过程中，随机让 40% 的神经元“罢工”（置零），防止过拟合
            nn.Linear(1024, 1024),#|
            nn.ReLU(),
            nn.Dropout(0.4),#隐藏层，保持高维度的情况下，进一步混合特征信息，加深网络的推理深度
            nn.Linear(1024, 256),#开始收缩信息，特征压缩。强迫模型丢弃不重要的信息，只保留最核心的 256 个特征用于最后的判断。这通常被称为“瓶颈（Bottleneck）”设计
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)#输出维度256，2个类别，二分类任务，score_0: 代表“无关联/负样本”的分数，score_1: 代表“有关联/正样本”的分数
        )
          
        # 初始化SAITS模型 -（Self-Attention-based Imputation for Time Series）处理时间序列数据中的缺失值（插补/Imputation）
        #处理药物特征序列中的缺失值
        self.saits_drug = SAITS(
            n_groups=args.n_groups,#多个组，利用这些组来迭代地细化填补的结果（第一组粗填，第二组精修...）
            n_group_inner_layers=args.n_group_inner_layers,#每个“组”内部包含多少层 Transformer Encoder
            d_feature=args.d_feature,#输入特征的维度，模型输入层的宽度
            d_model=args.d_model,#模型内部的隐藏层维度，Transformer 内部处理数据时的向量长度
            d_inner=args.d_inner,#前馈神经网络 (FFN) 的中间层维度，在 Transformer 的每个 Block 里都有一个 FeedForward 层，这个参数定义了该层“膨胀”得有多宽（通常是 d_model 的 4 倍）
            n_head=args.n_head,#多头注意力的头数
            d_k=args.d_k,#d_k (Key 维度)
            d_v=args.d_v,#d_v (Value 维度)
            dropout=args.saits_dropout,#防止过拟合的丢弃率
            input_with_mask=args.input_with_mask,#输入是否包含掩码 (Mask)
            param_sharing_strategy=args.param_sharing_strategy,#参数共享策略，SAITS 的不同“组（Groups）”之间是否共享权重？
            MIT=args.MIT,#掩码插补任务训练目标，通常是一个布尔值或权重值，模型学会填补缺失值
            device=device#指定模型运行在 CPU 还是 GPU 上
        )
        """输入：
              含有缺失值或噪声的药物特征序列。
        SAITS 处理：
              利用自注意力机制，根据上下文（Context）推测缺失的部分。
              利用 MIT 任务，学习数据内部的深层依赖关系。
        输出：
              完整的、高质量的药物特征表示。"""
        
        #处理疾病特征序列中的缺失值
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
        # === 修改部分：调用 Hypergraph Transformer ===
        # 原代码：dr_sim = self.gt_drug(drdr_graph, H_drdr)
        # 注意：HyperGT 主要依赖特征矩阵和超图关联矩阵 H，不需要 DGL 图对象 drdr_graph
        
        # 1. 确保 H 矩阵在设备上
        H_drdr = H_drdr.to(device)
        H_didi = H_didi.to(device)
        
        # 2. 如果 drug_feature 的维度不是 gt_out_dim，可能需要预处理
        # 假设这里 drug_feature 和 disease_feature 已经是适合输入的维度，或者在 HyperGT 内部会被 input_fc 映射
        # 根据原代码逻辑，drug_linear 是后面用的，所以这里传入原始特征（或者你需要先映射一下）
        # 通常 HyperGT 需要特征维度和 hidden_dim 匹配
        # 建议：如果 drug_feature 是 300 维，HyperGT 的 in_dim 设为 300，out_dim 设为 args.gt_out_dim
        
        dr_sim = self.gt_drug(drug_feature, H_drdr)
        di_sim = self.gt_disease(disease_feature, H_didi)
        # === 修改结束 ===
        
        # # 2. 线性变换对齐维度-将原始特征（可能维度不一）投影到统一的维度空间
        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)
        # # 3. 准备异构图数据，将三种不同类型的节点特征打包
        # feature_dict = {
        #     'drug': drug_feature,
        #     'disease': disease_feature,
        #     'protein': protein_feature
        # }

        # drdipr_graph.ndata['h'] = feature_dict
        # #利用 dgl.to_homogeneous 将复杂的异构图转换成一个同构的大图 g，并将所有特征拼接到一个长矩阵 feature 中，为接下来的 HGT（异构图 Transformer）做数据格式准备
        # g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        # feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)
        # #4. HGT 异构图卷积，提取异质性特征。让药物“看到”它作用的靶标蛋白，让疾病“看到”它关联的药物。这是全局信息的交互
        # for layer in self.hgt:
        #     hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
        #     feature = hgt_out
        # #解释：切片操作。从 HGT 输出的大矩阵中，把属于“药物”和“疾病”的行分别取出来。
        # dr_hgt = hgt_out[:self.args.drug_number, :]
        # di_hgt = hgt_out[self.args.drug_number:self.args.disease_number + self.args.drug_number, :]


         # === 3. 修改：数据准备与模型调用 ===
        
        # (1) 准备特征字典 x_dict
        x_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }
        
        # (2) 将 DGL 图转为超边索引 (利用我们写的工具函数)
        # 注意：drdipr_graph 必须包含 'drug', 'disease', 'protein' 节点
        hyperedge_index_dict = dgl_hetero_to_hyperedge_index(drdipr_graph, device)
        
        # (3) 调用 HHGT
        out_dict = self.hhgt(x_dict, hyperedge_index_dict)
        
        # (4) 提取输出，替换原本的 dr_hgt, di_hgt
        # 原代码：dr_hgt = hgt_out[:self.args.drug_number, :]
        # 新代码：直接从字典取
        dr_hgt = out_dict['drug']
        di_hgt = out_dict['disease']
        
        # === 修改结束 (后续代码保持不变) ===
        #【关键修复】投影对齐维度 (64 -> 200)
        dr_hgt = self.hhgt_out_proj(dr_hgt) # [Batch, 200]
        di_hgt = self.hhgt_out_proj(di_hgt) # [Batch, 200] 


        #特征融合，将“相似性特征”和“异构图特征”结合起来，堆叠张量。形状变为 (Batch, 2, Dim)
        #特征堆叠与融合
        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)
        #输入 Transformer Encoder-注意力融合。模型会自动学习这两个视角的特征哪个更重要，并生成更强的融合特征
        dr = self.drug_trans(dr)
        di = self.disease_trans(di)
        #展平，将 (Batch, 2, Dim) 变成 (Batch, 2*Dim)-准备进入下一阶段的特征处理
        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        # 生成掩码
        #drug_mask = torch.ones_like(dr).to(device)
        #disease_mask = torch.ones_like(di).to(device)
        # 药物特征掩码生成
        # 原始缺失掩码（0=缺失，1=观测）
        M_drug_original = (dr != 0).float().to(device)  # 假设0表示缺失
        # 人工缺失掩码（随机掩盖20%的观测值，用于MIT任务）
        #在原始存在的非零数据中，随机选出 20% 的位置，标记为 1（意思是：这里我要挖个坑让模型填）
        I_drug = torch.rand_like(dr) < 0.2  # 随机生成掩码
        I_drug = I_drug.float() * M_drug_original  # 仅在原始观测位置掩码
        # 总掩码（保留的观测值，用于ORT任务：1=保留，0=原始缺失+人工缺失）
        M_hat_drug = M_drug_original - I_drug

        # 3.2 疾病特征掩码生成（同理）
        M_di_original = (di != 0).float().to(device)
        I_di = torch.rand_like(di) < 0.2
        I_di = I_di.float() * M_di_original
        M_hat_di = M_di_original - I_di

        # 药物输入：仅保留总掩码标记的位置（人工缺失已被掩盖）
        #利用 SAITS 模型对特征进行修复和去噪
        #构造输入字典。将特征增加一个维度变成 3D (Batch, Sequence=1, Dim)，并根据掩码将那 20% 的数据强制置为 0
        dr_3d = dr.unsqueeze(1)  # (drug_num, 1, 2*gt_out_dim)
        dr_saits_input = dr_3d * M_hat_drug.unsqueeze(1)  # 施加人工缺失
        drug_inputs = {"X": dr_saits_input, "missing_mask": M_hat_drug.unsqueeze(1)}

        # 疾病输入：同理
        di_3d = di.unsqueeze(1)
        di_saits_input = di_3d * M_hat_di.unsqueeze(1)
        disease_inputs = {"X": di_saits_input, "missing_mask": M_hat_di.unsqueeze(1)}

        # 5. SAITS优化（必须使用修改后的SAITS，返回中间结果X_tilde_1/2/3）
        drug_saits_out = self.saits_drug.impute(drug_inputs)
        dr_saits = drug_saits_out[0]  # 最终插补结果
        X_tilde1_dr, X_tilde2_dr, X_tilde3_dr = drug_saits_out[1]  # 中间结果1,2,3


        disease_saits_out = self.saits_disease.impute(disease_inputs)
        di_saits = disease_saits_out[0]  # 最终插补结果
        X_tilde1_di, X_tilde2_di, X_tilde3_di = disease_saits_out[1]  # 中间结果1,2,3

        # 移除序列维度，去掉多余的维度，变回 2D 向量
        dr_saits = dr_saits.squeeze(1)
        di_saits = di_saits.squeeze(1)

        # 6.1 药物特征的MIT+ORT损失
        #损失计算与最终预测
        # MIT损失：人工缺失位置的插补误差（用I_drug掩码）
        #MIT Loss (填空题)。计算在被人工挖掉的 20% 位置上，模型填补的值与真实值 dr 的误差
        mit_loss_drug = masked_mae_cal(dr_saits, dr, I_drug)
        # ORT损失：保留观测位置的重建误差（用M_hat_drug掩码，取三个中间结果的平均）
        #ORT Loss (重建题)。计算在保留的 80% 位置上，模型重建的值与真实值的误差。使用 3 个中间层的输出来计算，保证模型训练的稳定性
        ort_loss_drug = (masked_mae_cal(X_tilde1_dr.squeeze(1), dr, M_hat_drug) +
                         masked_mae_cal(X_tilde2_dr.squeeze(1), dr, M_hat_drug) +
                         masked_mae_cal(X_tilde3_dr.squeeze(1), dr, M_hat_drug)) / 3

        # 6.2 疾病特征的MIT+ORT损失（同理）
        mit_loss_di = masked_mae_cal(di_saits, di, I_di)
        ort_loss_di = (masked_mae_cal(X_tilde1_di.squeeze(1), di, M_hat_di) +
                       masked_mae_cal(X_tilde2_di.squeeze(1), di, M_hat_di) +
                       masked_mae_cal(X_tilde3_di.squeeze(1), di, M_hat_di)) / 3

        # 6.1 药物SAITS的总损失（自身MIT+ORT联合优化）-药物通道的总自监督损失
        drug_total_loss = mit_loss_drug + ort_loss_drug  # 符合SAITS论文对单个模型的联合优化要求
        # 6.2 疾病SAITS的总损失（自身MIT+ORT联合优化）
        disease_total_loss = mit_loss_di + ort_loss_di  # 同理
        #特征交互，sample[:, 0/1]：取出当前 Batch 中成对的药物 ID 和 疾病 ID
        #torch.mul：哈达玛积（元素对应相乘）。这是计算两个向量相似度或交互作用的常用方法
        drdi_embedding = torch.mul(dr_saits[sample[:, 0]], di_saits[sample[:, 1]])
        #通过多层感知机（MLP）输出最终的预测分数（如：0.8 表示有关联）
        output = self.mlp(drdi_embedding)

        # 返回：药物总损失、疾病总损失、预测结果（供分别训练）
        #这三个值会在外部的训练循环中加权求和，进行反向传播
        return drug_total_loss, disease_total_loss, output