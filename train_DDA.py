import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
from data_preprocess import *
from model.AMNTDDA import AMNTDDA
from metric import *
import scipy.sparse as sp
from dgl import DGLGraph
from collections import defaultdict
from scipy.sparse import csr_matrix
from modeling.saits import SAITS
from modeling.loss_functions import mit_loss, ort_loss
from torch.utils.data import DataLoader, TensorDataset
from modeling.utils import masked_mae_cal
from hypergraph_utils import construct_hypergraph,Multi_omics_hyperedge_concat,construct_H_with_KNN_from_distance,generate_G_from_H
import sklearn.metrics as metrics
from datetime import datetime
import sys  # <--- 必须有这一行
# from DualHyperGT import DualHyperGT

device = torch.device('cuda')

# === 新增：日志记录类 ===
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout  # 屏幕输出流
        self.log = open(filename, "a", encoding='utf-8')  # 文件输出流
    
    def write(self, message):
        self.terminal.write(message)  # 打印到屏幕
        self.log.write(message)       # 写入到文件
        self.log.flush()              # 立即刷新缓冲区，防止程序崩溃丢失日志
    
    def flush(self):
        pass


def train(args, model, train_loader, optimizer, epoch,
          drdr_graph, didi_graph, drdipr_graph,
          drug_feature, disease_feature, protein_feature,
          H_drdr, H_didi):
    model.train()
    #total_cls_loss = 0.0  # 分类损失（主任务）
    total_drug_reg_loss = 0.0  # 药物SAITS回归损失（辅助）
    total_disease_reg_loss = 0.0  # 疾病SAITS回归损失（辅助）
    #total_acc = 0.0  # 分类准确率（辅助监控）

    # 分类损失函数（主任务）
    cls_criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(train_loader):
        # 解析batch数据：样本索引、分类标签
        sample = batch[0].to(device)  # 样本索引 (batch_size, 2)
        y_cls = batch[1].to(device).squeeze()  # 压缩维度

        # 前向传播：获取辅助回归损失和分类输出
        drug_reg_loss, disease_reg_loss, cls_output = model(
            drdr_graph, didi_graph, drdipr_graph,
            drug_feature, disease_feature, protein_feature,
            sample, H_drdr, H_didi
        )

        # 计算主任务分类损失
        cls_loss = cls_criterion(cls_output, y_cls)

        # 联合优化总损失 = 分类损失 + 辅助回归损失
        total_loss = cls_loss + drug_reg_loss + disease_reg_loss
        # total_loss = drug_reg_loss + disease_reg_loss

        # 反向传播与参数更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 累计损失
        #total_cls_loss += cls_loss.item()
        total_drug_reg_loss += drug_reg_loss.item()
        total_disease_reg_loss += disease_reg_loss.item()

        # 计算分类准确率（仅用于监控）
        #pred_cls = torch.argmax(cls_output, dim=-1)  # 预测类别
        #acc = (pred_cls == y_cls).float().mean()
        #total_acc += acc.item()

        # 打印批次信息
        if (batch_idx + 1) % 10 == 0:
            print(f"Train Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, "
                 # f"分类损失: {cls_loss.item():.4f}, "
                  f"药物回归损失: {drug_reg_loss.item():.4f}, "
                  f"疾病回归损失: {disease_reg_loss.item():.4f}, ")
                  #f"准确率: {acc.item():.4f}")

    # 计算平均指标
    #avg_cls_loss = total_cls_loss / len(train_loader)
    avg_drug_reg_loss = total_drug_reg_loss / len(train_loader)
    avg_disease_reg_loss = total_disease_reg_loss / len(train_loader)
    #avg_acc = total_acc / len(train_loader)
    
    avg_total_loss = avg_drug_reg_loss + avg_disease_reg_loss

    print(f"\n本轮训练结束：")
    print(#f"平均分类损失: {avg_cls_loss:.4f}, "
          f"平均药物回归损失: {avg_drug_reg_loss:.4f}, "
          f"平均疾病回归损失: {avg_disease_reg_loss:.4f}, ")
          #f"平均准确率: {avg_acc:.4f}")

    return avg_total_loss, 0.0#, avg_cls_loss  # 返回总损失和主任务分类损失（用于早停）


def validate(args, model, val_loader,
             drdr_graph, didi_graph, drdipr_graph,
             drug_feature, disease_feature, protein_feature,
             H_drdr, H_didi):
    model.eval()
    #total_cls_loss = 0.0
    total_drug_reg_loss = 0.0
    total_disease_reg_loss = 0.0
    #total_acc = 0.0

    # 存储所有预测概率和标签（用于计算AUC、AUPR）
    all_pred_probs = []
    all_labels = []
    #验证集也要计算分类损失 
    cls_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in val_loader:
            sample = batch[0].to(device) # 样本索引
            y_cls = batch[1].to(device).squeeze()  # 新增 .squeeze()
            labels_np = y_cls.cpu().numpy()  # 转成numpy存真实标签
            all_labels.extend(labels_np)  # 累计所有真实标签

            # 前向传播
            drug_reg_loss, disease_reg_loss, cls_output = model(
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                sample, H_drdr, H_didi
            )

            # 计算损失
            #cls_loss = cls_criterion(cls_output, y_cls)
            #total_cls_loss += cls_loss.item()
            total_drug_reg_loss += drug_reg_loss.item()
            total_disease_reg_loss += disease_reg_loss.item()

            # 计算准确率
            #pred_cls = torch.argmax(cls_output, dim=-1)
            #acc = (pred_cls == y_cls).float().mean()
            #total_acc += acc.item()

            # 保存正类预测概率（用于AUC计算）
            #pred_probs = torch.softmax(cls_output, dim=-1)[:, 1].cpu().numpy()
            #all_pred_probs.extend(pred_probs)


    # 计算平均损失和准确率
    #avg_cls_loss = total_cls_loss / len(val_loader)
    avg_drug_reg_loss = total_drug_reg_loss / len(val_loader)
    avg_disease_reg_loss = total_disease_reg_loss / len(val_loader)
    #avg_acc = total_acc / len(val_loader)

    # 1. 根据预测概率生成预测类别（以0.5为阈值，概率>=0.5视为正类）
    #pred_classes = (np.array(all_pred_probs) >= 0.5).astype(int)  # 转换为0/1的整数类别
    # 2. 调用get_metric，传入三个参数（与函数定义一致）
   # AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(
   #     all_labels,  # 第一个参数：真实标签（y_true）
   #    pred_classes,  # 第二个参数：预测类别（y_pred）
   #     all_pred_probs  # 第三个参数：预测概率（y_prob）
#)

    # 打印验证集结果
    print(f"\n验证集：")
    print(#f"平均分类损失: {avg_cls_loss:.4f}, "
          f"平均药物回归损失: {avg_drug_reg_loss:.4f}, "
          f"平均疾病回归损失: {avg_disease_reg_loss:.4f}, ")
          #f"准确率: {avg_acc:.4f}")
    #print(f"AUC: {AUC:.4f}, AUPR: {AUPR:.4f}, "
    #      f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
    #      f"F1-score: {f1:.4f}, MCC: {mcc:.4f}")

    total_val_loss = avg_drug_reg_loss + avg_disease_reg_loss
    return total_val_loss, 0.0#, avg_cls_loss  # 返回loss, mae


#超图构建模块
# 传入图对象,根据成对边和k阶邻居生成超图,返回超图的节点集合和超边集合
# def generate_hypergraph_matrix(graph, k=1):
#     """
#     根据成对边和k阶邻居生成超图。

#     :param graph: DGLGraph 对象
#     :param k: 邻居阶数,默认为1
#     :return: 超图的节点集合和超边集合
#     """
#     """
#     - 输入：普通图 (node1-node2 这样的成对边)
#     - 输出：超图矩阵 H (节点×超边的关联矩阵)
#     - 普通图的边：只能连接2个节点，如 (drug1, drug2)
#     - 超边：可以连接多个节点，如 (drug1, drug2, drug3, drug4)

#     参数:
#     - graph: DGL图对象 (例如药物相似性图drdr_graph)
#     - k: 邻居阶数 (k=1表示只考虑直接邻居)

#     返回:
#     - H矩阵: torch稀疏张量 [N_nodes × N_hyperedges]
#     """
#      # === 第1步: 提取基础信息 ===
#     # 获取节点数量和边列表
#     num_nodes = graph.number_of_nodes()
#     #提取所有边，转换为列表 [(node1, node2), (node3, node4),
#     #zip(*graph.edges()) 是Python的解包技巧，将边的起点和终点分别解包成两个列表
#     #graph.edges() 返回 ([源节点列表], [目标节点列表])
#     #zip(*...) 将它们配对-例如: [(drug1, drug5), (drug2, drug8), ...]
#     edges = list(zip(*graph.edges()))

#     # 初始化超边集合
#     # === 第2步: 初始化超边集合 ===
#     hyperedges = set(edges)  # 成对边作为初始超边
#      # 📌 将普通边作为初始超边
#     # 💡 关键点: 每条二元边 (u, v) 也是一个超边！
#     # 例如: {(0,1), (0,5), (1,3), ...}
#     # 此时每个超边只包含2个节点

#     # 使用 BFS 或 DFS 来找到 k 阶邻居
#     # === 第3步: 构建邻居字典 ===
#     # 使用 BFS 或 DFS 来找到 k 阶邻居
#     # === 第3步: 构建邻居字典 ===
#     neighbors = defaultdict(set)
#     # 📌 创建字典存储每个节点的邻居
#     # defaultdict(set) 表示默认值是空集合
#     for u, v in edges:
#         # 遍历每条边
#         neighbors[u].add(v) # u的邻居中加入v
#         neighbors[v].add(u) # v的邻居中加入u (无向图)

#     #=== 第4步: 扩展到k阶邻居 (核心!) ===
#     for _ in range(k - 1):
#         new_neighbors = defaultdict(set)
#         for node in range(num_nodes):
#             for neighbor in neighbors[node]:
#                 new_neighbors[node] |= neighbors[neighbor]
#             new_neighbors[node] -= {node}  # 去除自身
#         neighbors = new_neighbors

#     # 将 k 阶邻居添加到超边集合中
#     for node, neighbor_set in neighbors.items():
#         if len(neighbor_set) > 0:
#             hyperedges.add(tuple(sorted([node] + list(neighbor_set))))

#     H = build_hypergraph_matrix(num_nodes, hyperedges)
#     return csr_to_sparse_tensor(H)  # 返回torch稀疏张量


# 通过节点数量和超边集合,构建超图的关联矩阵H,返回的是关联矩阵H
def build_hypergraph_matrix(num_nodes, hyperedges):
    """
    构建超图的关联矩阵 H。

    :param num_nodes: 节点数量
    :param hyperedges: 超边集合
    :return: 关联矩阵 H (稀疏矩阵)
    """
    rows = []
    cols = []
    for i, hyperedge in enumerate(hyperedges):
        for node in hyperedge:
            rows.append(node)
            cols.append(i)

    data = np.ones(len(rows))
    # 构建这种(压缩稀疏行矩阵)，通过指定非零元素的值、非零元素所在的行索引以及列索引来创建一个稀疏矩阵(data, (rows, cols))。
    H = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, len(hyperedges)))
    return H


# 将 scipy.sparse.csr_matrix 转换为 torch.sparse_coo_tensor
def csr_to_sparse_tensor(csr_matrix):
    """
    :param csr_matrix: scipy.sparse.csr_matrix 对象
    :return: torch.sparse_coo_tensor 对象
    把scipy中的稀疏矩阵转换成torch中的稀疏张量,提升效率
    """
    # 获取稀疏矩阵的非零元素的位置和值
    coo = csr_matrix.tocoo()
    values = torch.tensor(coo.data, dtype=torch.float)
    #  indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
    # 创建 torch.sparse_coo_tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size=csr_matrix.shape)
    return sparse_tensor


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--k_fold', type=int, default=10, help='k-fold cross validation')  #K折交叉验证 (默认: 10)
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')#1000->300 #训练轮数 (默认: 10)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') #学习率 (Learning Rate, 默认: 0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight_decay')#权重衰减 (默认: 0.001)
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')# 随机种子 (默认: 1234)
    parser.add_argument('--neighbor', type=int, default=20, help='neighbor') #邻居数量 (默认: 20)
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate') #负采样率 (默认: 1.0)
    parser.add_argument('--dataset', default='C-dataset', help='dataset') #数据集名称 (默认: 'C-dataset')
    #  parser.add_argument('--dropout', default='0.2', type=float, help='dropout')
    #Graph Transformer (GT) 参数
    parser.add_argument('--gt_layer', default='2', type=int, help='graph transformer layer')#层数 (默认: 2)
    parser.add_argument('--gt_head', default='2', type=int, help='graph transformer head')#注意力头数 (默认: 2)
    parser.add_argument('--gt_out_dim', default='200', type=int, help='graph transformer output dimension')#输出维度 (默认: 200)
    #Heterogeneous Graph Transformer (HGT) 参数
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')#层数 (默认: 2)
    parser.add_argument('--hgt_head', default='8', type=int, help='heterogeneous graph transformer head')#注意力头数 (默认: 8)
    parser.add_argument('--hgt_in_dim', default='64', type=int, help='heterogeneous graph transformer input dimension')#输入维度 (默认: 64)
    parser.add_argument('--hgt_head_dim', default='25', type=int, help='heterogeneous graph transformer head dimension')#每个头的维度 (默认: 25)
    parser.add_argument('--hgt_out_dim', default='200', type=int,
                        help='heterogeneous graph transformer output dimension')#输出维度 (默认: 200)
    #Transformer (Tr) 参数     
    parser.add_argument('--tr_layer', default='2', type=int, help='transformer layer')#层数 (默认: 2)
    parser.add_argument('--tr_head', default='4', type=int, help='transformer head')#注意力头数 (默认: 4)

    # 添加SAITS子模块的参数,SAITS 是一个专门用于处理序列数据缺失值（插补）的 Transformer 架构模型。
    parser.add_argument("--n_groups", type=int, default=2,
                        help="SAITS 分组数（控制模型并行结构，影响特征分组学习）")#分组数 (默认: 2)
    parser.add_argument("--n_group_inner_layers", type=int, default=2,
                        help="每组内的 Transformer 层数（决定特征交互深度）")#每组内的 Transformer 层数 (默认: 2)
    #  parser.add_argument("--d_feature", type=int, default=128,
    #                    help="输入特征维度（需与 AMDGT 输出特征维度匹配")

    parser.add_argument("--d_feature", type=int, default=400,
                        help="输入特征维度（需与 AMDGT 输出特征维度匹配")

    #  parser.add_argument("--d_model", type=int, default=128,
    #                    help="Transformer 模块的隐藏维度（需与特征维度适配）")

    parser.add_argument("--d_model", type=int, default=400,
                        help="Transformer 模块的隐藏维度（需与特征维度适配）")

    parser.add_argument("--d_inner", type=int, default=512,
                        help="Transformer 前馈网络的隐藏维度（影响非线性变换能力）")
    parser.add_argument("--n_head", type=int, default=4,
                        help="Multi-Head Attention 的头数（控制注意力并行度）")

    # parser.add_argument("--d_k", type=int, default=32,
    #                    help="每个注意力头的 key/value 维度（需满足 d_model = n_head * d_k）")
    # parser.add_argument("--d_v", type=int, default=32,
    #                    help="每个注意力头的 value 维度（同 d_k 逻辑）")

    parser.add_argument("--d_k", type=int, default=100,
                        help="每个注意力头的 key/value 维度（需满足 d_model = n_head * d_k）")
    parser.add_argument("--d_v", type=int, default=100,
                        help="每个注意力头的 value 维度（同 d_k 逻辑）")

    # SAITS 训练与任务参数
    #  parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--input_with_mask", action="store_true", default=True)#是否将“掩码矩阵”（Mask）作为输入的一部分喂给模型。True (默认)：模型看到数据 [0.5, 0, 0.8] + 掩码 [1, 0, 1]。
    parser.add_argument("--param_sharing_strategy", type=str, default="within_group",
                        choices=["within_group", "between_group"])#（组内共享，组间共享）权重
    parser.add_argument("--MIT", action="store_true", default=True)#掩码插补任务，SAITS 能够“自我学习”的核心开关
    # 分别为 AMDGT 和 SAITS 设置 dropout
    parser.add_argument('--amdgt_dropout', type=float, default=0.2,
                        help='dropout rate for AMDGT components')#主模型 AMDGT（图神经网络部分，GT/HGT），随机丢弃率
    parser.add_argument('--saits_dropout', type=float, default=0.1,
                        help='dropout rate for SAITS components')# 子模块 SAITS（Transformer部分）

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    args.result_dir = 'Result/' + args.dataset + '/AMNTDDA/'
    os.makedirs(args.result_dir, exist_ok=True)

    #  # === 启用日志记录 ===
    # # 1. 获取当前时间，格式如：2023-10-27_15-30-00
    # # current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # # # 2. 定义日志文件名
    # # log_filename = os.path.join(args.result_dir, f'result_{current_time_str}.txt')
    # # # 3. 重定向 print 输出
    # # # 从这一行开始，所有的 print() 都会同时显示在屏幕和写入 txt 文件
    # # sys.stdout = Logger(log_filename)
    # # === 【修改 2】: 手动打开一个文件用于保存结果 ===
    # current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # result_file_path = os.path.join(args.result_dir, f'result_{current_time_str}.txt')
    # result_file = open(result_file_path, 'a', encoding='utf-8')
    # # 定义一个辅助函数，专门用于同时打印到屏幕 并 写入文件
    # def log_result(content):
    #     print(content)  # 打印到屏幕
    #     result_file.write(content + '\n') # 写入文件
    #     result_file.flush() # 立即保存
    # print(f"日志功能已启动，结果将保存至: {result_file_path}")
    # print(f"当前运行参数: {args}")

    # -------------------------------------------------------
    # 1. 准备日志文件
    # -------------------------------------------------------
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
        
    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 定义日志文件路径
    log_file_path = os.path.join(args.result_dir, f'result_{current_time_str}.txt')
    
    # 打开文件对象 (追加模式 'a')
    f_log = open(log_file_path, 'a', encoding='utf-8')

    # 定义一个专用函数：既打印到屏幕，又写入文件
    def log_msg(content):
        print(content)  # 屏幕显示
        f_log.write(str(content) + '\n') # 写入文件
        f_log.flush()   # 立即保存，防止程序崩溃丢失数据

    # 记录必要的启动信息
    log_msg(f"Current Date: {datetime.now().strftime('%Y-%m-%d')}")
    log_msg(f"Log file path: {log_file_path}")
    log_msg(f"Parameters: {args}")
    # -------------------------------------------------------
    # 2. 加载数据 (这一步可能比较慢，请耐心等待)
    # -------------------------------------------------------
    print("正在加载数据，请稍候...")  # 这行不需要进日志，用普通 print
    
    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    data = k_fold(data, args)
    # print("\n" + "="*30 + " Data 结构概览 " + "="*30)
    # for key, value in data.items():
    #     # 情况1: 如果是 PyTorch Tensor 或 Numpy 数组，打印形状
    #     if hasattr(value, 'shape'): 
    #         print(f"[{key:<15}]: Shape {value.shape} | Type: {type(value).__name__}")
        
    #     # 情况2: 如果是 List (通常存 K-Fold 的索引)，打印长度和前几个元素
    #     elif isinstance(value, list):
    #         print(f"[{key:<15}]: List Len {len(value)} | First item: {value[0] if len(value)>0 else 'Empty'}")
        
    #     # 情况3: 其他数值或字符串
    #     else:
    #         print(f"[{key:<15}]: {value}")
    # print("="*75 + "\n")
    #drdr_graph 药物同构图   didi_graph (疾病同构图)
    drdr_graph, didi_graph, data = dgl_similarity_graph(data, args)
    # ================= 关键修改：先定义特征变量 =================
      #准备公式7的X输入h^(0) = W^0 xi+ b
    drug_feature = torch.FloatTensor(data['drugfeature']).to(device)# 嵌入特征
    disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)# 嵌入特征
    protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    all_sample = torch.tensor(data['all_drdi']).long()

    #-----add- dwp 2026-1-28---
    features_drug_struct = torch.FloatTensor(data['drf']).to(device)      # 结构特征
    features_drug_func = torch.FloatTensor(data['drg']).to(device)         # 功能特征

    features_dis_pheno = torch.FloatTensor(data['dip']).to(device)         # 表型特征
    features_dis_func = torch.FloatTensor(data['dig']).to(device)          # 功能特征

    #--------------------------
    # 新增：为drdr_graph和didi_graph生成H矩阵
    # H_drdr = generate_hypergraph_matrix(drdr_graph, k=1).to(device)
    # H_didi = generate_hypergraph_matrix(didi_graph, k=1).to(device)

    # --- 新代码 ---
    # 注意：我们要用“特征”来构建，而不是用“图”来构建
    # K_neigs: 每一个节点找多少个邻居构成一个超边，建议设为 10-20
    # edge_type: 如果特征是连续数值，'euclid' 通常较好；如果是表达谱/指纹相似度，'pearson' 也可以尝试

    print("正在重新构建药物超图 (基于特征)...")
    # print("drug_feature=",{drug_feature})
    #  # === 加入这一行检查形状 ===
    # print(f"drug_feature SHAPE: {drug_feature.shape}") 
    # # ========================

    # H_drdr = construct_hypergraph(drug_feature, 
    #                             K_neigs=[15],       # 参数可调，scMHNN用了70，但对于小数据集建议 10-15
    #                             is_probH=True, 
    #                             m_prob=1.5,         # 开启概率权重，效果通常优于 0/1
    #                             edge_type='euclid') # 或 'pearson'
    # H_drdr = H_drdr.to(device) # 别忘了放到 GPU 上

    # print("正在重新构建疾病超图 (基于特征)...")
    # H_didi = construct_hypergraph(disease_feature, 
    #                             K_neigs=[15], 
    #                             is_probH=True, 
    #                             m_prob=1.5,     
    #                             edge_type='euclid')
    # H_didi = H_didi.to(device)
    

   # === 2. 构建药物多模态超图 ===
   # 结构超图 (H_struct)
    H_drug_1 = construct_hypergraph(features_drug_struct, K_neigs=[10],is_probH=True,  m_prob=1.5, edge_type='pearson')
    # 功能超图 (H_func)
    H_drug_2 = construct_hypergraph(features_drug_func, K_neigs=[10],is_probH=True,  m_prob=1.5, edge_type='euclid') 
    # 语义超图 (H_embed)
    H_drug_3 = construct_hypergraph(drug_feature, K_neigs=[10],is_probH=True,  m_prob=1.5, edge_type='euclid')

    # 融合药物超图 (拼接列)
    # H_drug 维度: [N_drug, (N_drug*3)]
    H_drug_final = Multi_omics_hyperedge_concat(H_drug_1, H_drug_2, H_drug_3)
     # 【关键】移动到 GPU
    H_drug_final = H_drug_final.to(device) 

    # === 3. 构建疾病多模态超图 ===
    H_dis_1 = construct_hypergraph(features_dis_pheno, K_neigs=[10],is_probH=True,  m_prob=1.5, edge_type='pearson')
    H_dis_2 = construct_hypergraph(features_dis_func, K_neigs=[10], is_probH=True,  m_prob=1.5,edge_type='euclid')
    H_dis_3 = construct_hypergraph(disease_feature, K_neigs=[10], is_probH=True,  m_prob=1.5,edge_type='euclid')

    H_dis_final = Multi_omics_hyperedge_concat(H_dis_1, H_dis_2, H_dis_3)
    # 【关键】移动到 GPU
    H_dis_final = H_dis_final.to(device)
    # # === 4. 生成 G 矩阵供模型使用 ===
    # G_drug = generate_G_from_H(H_drug_final)
    # G_dis = generate_G_from_H(H_dis_final)
    

    # 打印 H_drdr 前 10 行，前 10 列的数值
    # print("=== H_drdr 前 10x10 区域数值 ===")
    # to_dense() 将稀疏矩阵转为普通矩阵
    # print(H_drdr.to_dense()[:10, :10]) 
    # print(H_drdr)
    # 检查是否包含权重（如果是概率超图，值应该是小数；如果是二值超图，值是0或1）
    # print("\n=== H_drdr 样本值检查 ===")
    # 打印前20个非零元素的值
    # print(H_drdr.values[:50])
    # # 打印 H_drdr 中所有非0且非1的独特值
    # unique_values = torch.unique(H_drdr.values())
    # print("Unique weights in H:", unique_values)
    # 验证代码
    # vals = H_drug_final.values()
    # print(f"最大权重: {vals.max().item():.4f}") # 应该是 1.0 (自环)
    # print(f"最小权重: {vals.min().item():.4f}") # 应该明显小于 1.0 (比如 0.5, 0.7 等)
    # print(f"平均权重: {vals.mean().item():.4f}") # 应该在 0.8-0.9 左右
    # print(f"前 50 个值: {vals[:50]}")

    # 打印统计信息 (调试用)
    print(f"药物融合超图维度: {H_drug_final.shape}, 边数: {H_drug_final._nnz()}")
    print(f"疾病融合超图维度: {H_dis_final.shape}, 边数: {H_dis_final._nnz()}")
    # 提取非零权值
    # 1. 显式合并（这是解决报错的关键）
    H_drug_final = H_drug_final.coalesce()
    vals = H_drug_final.values()

    if vals.numel() > 0:
        print(f"--- 融合超图统计 ---")
        print(f"超边总数: {H_drug_final.shape[1]}")
        print(f"最大权重: {vals.max().item():.4f}") 
        print(f"最小权重: {vals.min().item():.4f}") 
        print(f"平均权重: {vals.mean().item():.4f}") 
        # 打印前50个，带点格式看起来不乱
        print(f"前 50 个权重值:\n{vals[:50].tolist()}")
    else:
        print("警告: H_drug_final 中没有非零值！")
#   # 1. 打印前 50 个数值 (扁平化)
#     print("G_drug 前 50 个数值:")
#     print(G_drug.view(-1)[:50]) 

#     # 2. 或者查看统计信息 (推荐，更能看出矩阵是否有问题)
#     print(f"G_drug 形状: {G_drug.shape}")
#     print(f"G_drug 最大值: {G_drug.max().item()}")
#     print(f"G_drug 最小值: {G_drug.min().item()}")
#     print(f"G_drug 平均值: {G_drug.mean().item()}")
# ----------------
    #------------------------------------------------------------
    drdr_graph = drdr_graph.to(device)
    didi_graph = didi_graph.to(device)
    # print("第一个关联矩阵是")
    # print(H_drdr)
    # print("第二个关联矩阵是")
    # print(H_didi)
    """
    第一个关联矩阵是
    tensor(indices=tensor([[    0,     0,     0,  ...,   662,   662,   662],
                        [  790,  1121,  1364,  ..., 33497, 33690, 33812]]),
        values=tensor([1., 1., 1.,  ..., 1., 1., 1.]),
       device='cuda:0', size=(663, 34155), nnz=66984, layout=torch.sparse_coo)
    第二个关联矩阵是
    tensor(indices=tensor([[    0,     0,     0,  ...,   408,   408,   408],
                        [  119,   324,   560,  ..., 21442, 21480, 21654]]),
        values=tensor([1., 1., 1.,  ..., 1., 1., 2.]),
        device='cuda:0', size=(409, 22232), nnz=43646, layout=torch.sparse_coo)
    """
    # #准备公式7的X输入h^(0) = W^0 xi+ b
    # drug_feature = torch.FloatTensor(data['drugfeature']).to(device)
    # disease_feature = torch.FloatTensor(data['diseasefeature']).to(device)
    # protein_feature = torch.FloatTensor(data['proteinfeature']).to(device)
    # all_sample = torch.tensor(data['all_drdi']).long()
    
    # print(torch.FloatTensor(data['drugfeature']))
    # print(torch.FloatTensor(data['diseasefeature']))
    # print(torch.FloatTensor(data['proteinfeature']))


    start = timeit.default_timer()#启动计时器

    cross_entropy = nn.CrossEntropyLoss()#定义损失函数-交叉熵损失
    log_file_path = os.path.join(args.result_dir, 'training_metrics.txt')
    # Metric = ('Epoch\t\tTime\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
  # === 3. 更新 Header 定义 ===
    # Metric_Header = "Epoch\t\tTime\t\tLL\t\tAcc\t\tRMSE\t\tMAE\t\tRecall\t\tPrec\t\tF1\t\tAUC\t\tAUPRC\t\tSpec\t\tBrier\t\tTP\t\tFN\t\tFP\t\tTN\t\tPosAvg\t\tNegAvg"
    Metric_Header = (
        f"{'Epoch':<6}{'Time':<8}{'LL':<10}{'Acc':<10}{'RMSE':<10}{'MAE':<10}"
        f"{'Recall':<10}{'Prec':<10}{'F1':<10}{'AUC':<10}{'AUPRC':<10}{'Spec':<10}"
        f"{'Brier':<10}{'TP':<6}{'FN':<6}{'FP':<6}{'TN':<6}{'PosAvg':<10}{'NegAvg':<10}"
    )
    AUCs, AUPRs = [], []

    # print('Dataset:', args.dataset)
    log_msg(f'Dataset: {args.dataset}')
    for i in range(args.k_fold):
        # 记录折数
        log_msg(f'\nFold: {i}')
        # 记录表头，方便后续复制到 Excel
        log_msg(Metric_Header)


        # print('fold:', i)
        # print(Metric)
        # print(Metric_Header) # 替换原来的 print(Metric)

    #  # === 初始化模型 (DualHyperGT) ===
    #     model = DualHyperGT(
    #         args=args,
    #         n_drug_nodes=drug_feature.shape[0],
    #         n_dis_nodes=disease_feature.shape[0],
    #         n_drug_hes=H_drug_final.shape[1],
    #         n_dis_hes=H_dis_final.shape[1],
    #         drug_in_channels=drug_feature.shape[1],
    #         dis_in_channels=disease_feature.shape[1],
    #         hidden_channels=64, # 可调整
    #         out_channels=64
    #     ).to(device)

        #model = AMNTDDA(args)
        #model = model.to(device)
        #optimizer = optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        # 初始化模型（假设AMNTDDA需要超图参数和SAITS参数）
        model = AMNTDDA(args).to(device)

        # 初始化优化器
        optimizer = optim.Adam(
            model.parameters(),#更新model的参数
            lr=args.lr,#学习率
            weight_decay=args.weight_decay#权重衰减
        )

        # 学习率调度器（可选，用于衰减学习率）
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=True
        )#factor=0.5 学习率砍半（乘以 0.5），verbose=True显示详细信息，如损失不下降时打印提示文字

        # 关键修改：创建当前折的训练集DataLoader
        X_train = torch.LongTensor(data['X_train'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        # 封装为TensorDataset
        #train_dataset = TensorDataset(X_train, Y_train)
        # 从完整训练集中划分20%作为验证集，80%作为实际训练集
        val_size = int(0.2 * len(X_train))  # 验证集大小
        X_val = X_train[:val_size]  # 验证集样本
        Y_val = Y_train[:val_size]
        X_train = X_train[val_size:]  # 实际训练集样本（剩余80%）
        Y_train = Y_train[val_size:]
        # 创建训练集DataLoader（使用划分后的训练数据）
        train_dataset = TensorDataset(X_train, Y_train)
        data['train_loader'] = DataLoader(
            train_dataset,
            batch_size=64,  # 可根据需求调整
            shuffle=True,# 【关键】每个Epoch开始时是否打乱数据？必须是 True！
            drop_last=False # 如果最后剩的数据不够64条，是否丢弃？False表示保留。
        )

        # 强制创建验证集DataLoader（不再依赖data中是否有'X_val'/'Y_val'）
        val_dataset = TensorDataset(X_val, Y_val)
        data['val_loader'] = DataLoader(
            val_dataset,
            batch_size=64, # 每次喂给模型 64 条数据
            shuffle=False, 
            drop_last=False # 如果最后剩的数据不够64条，是否丢弃？False表示保留。
        )

        # 准备训练集和验证集加载器（假设data包含k折划分的训练/验证数据）
        best_val_loss = float('inf')#设定为正无穷大
        best_model_path = os.path.join(args.result_dir, 'best_model.pth')#保存最佳训练模型
        #add dwp
        counter = 0
        patience = 20 # Early stopping patience
        
        best_metrics_str = ""  # 记录最佳AUC时的各个指标#add dwp
        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train = torch.LongTensor(data['X_train'][i]).to(device)#重新加载完整训练集
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)

        X_test = torch.LongTensor(data['X_test'][i]).to(device)#加载测试集的索引，搬运到 GPU
        Y_test = data['Y_test'][i].flatten()#压缩维度，Scikit-learn 计算 AUC/AUPR 的函数通常要求标签是一维数组
        # print(data['Y_test'][i])

        drdipr_graph, data = dgl_heterograph(data, data['X_train'][i], args)
        drdipr_graph = drdipr_graph.to(device)

        for epoch in range(args.epochs):
            #model.train()
            #train_loss, train_mae = train(args, model, data['train_loader'], optimizer, epoch)
            # 验证（假设data['val_loader']是当前折的验证集）
            #val_loss, val_mae = validate(args, model, data['val_loader'])
            train_loss, train_mae = train(
                args, model, data['train_loader'], optimizer, epoch,
                drdr_graph, didi_graph, drdipr_graph,  # 图数据
                drug_feature, disease_feature, protein_feature,  # 全局特征
                H_drug_final, H_dis_final # 传入新构建的超图
            )
            # 验证函数同样补充参数
            val_loss, val_mae = validate(
                args, model, data['val_loader'],
                drdr_graph, didi_graph, drdipr_graph,
                drug_feature, disease_feature, protein_feature,
                H_drug_final, H_dis_final
            )

            # 调整学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss
                }, best_model_path)
                print(f"Saved best model at epoch {epoch}, Val Loss: {best_val_loss:.4f}")
            else:
                counter += 1 # <--- 重要：如果没进步，计数器+1
                print(f"No improvement. Counter: {counter}/{patience}")
            
            # --- 3. 加入中断逻辑 ---
            if counter >= patience:
                # print(f"Early stopping triggered at epoch {epoch}!")
                log_msg(f"Early stopping triggered at epoch {epoch}!") # 建议把早停信息写入文件
                break  # <--- 重要：强制跳出循环


            with torch.no_grad():
                model.eval()# 1. 使用 test_logits 接收模型原始输出，绝对不要覆盖它
                d_loss, di_loss, test_raw_logits= model(drdr_graph, didi_graph, drdipr_graph, drug_feature,
                                                      disease_feature, protein_feature, X_test, H_drug_final, H_dis_final)

            
            # 3. 先计算概率 (Softmax 需要 Tensor 格式的 Logits)
            test_prob_tensor = fn.softmax(test_raw_logits, dim=-1)# 将得分转为概率 (0~1)
            # 4. 再计算预测类别 (Argmax)
            # test_score = torch.argmax(test_logits, dim=-1)#转类别
            test_pred_tensor = torch.argmax(test_raw_logits, dim=-1)#
            #  统一转为 Numpy，准备传给 get_metric
            # 取出正样本(索引1)的概率，并转到 CPU
            test_prob = test_prob_tensor[:, 1].cpu().numpy()# 只取“正样本(1)”的概率
            test_score = test_pred_tensor.cpu().numpy()
            # test_prob = test_prob.cpu().numpy()# 转为numpy数组

            # test_score = test_score.cpu().numpy()
            # # y_true_np: 真实标签 (确保它也是numpy数组)
            # y_true_np = Y_test

            # # 4. 统一转为 Numpy
            # # 取出正类(1)的概率
            # y_prob_np = test_prob_tensor[:, 1].cpu().numpy()
            # # 取出预测类别
            # y_pred_np = test_pred_tensor.cpu().numpy()
            # # 真实标签
            # y_true_np = Y_test
             
            # 4. 调用 metric.py 计算指标
            # AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)
            AUC, AUPR, accuracy, precision, recall, f1, mcc, ll, rmse, mae, specificity, brier, tp, fn_count, fp, tn, pos_avg, neg_avg = get_metric(Y_test, test_score, test_prob)

            end = timeit.default_timer()
            time = end - start
            # show = [epoch + 1, round(time, 2), round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
            #         round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            
            # 定义要打印的数据列表 (顺序需对应表头)
            # 表头: Epoch, LL, Acc, RMSE, MAE, Recall, Precision, F1, AUC, AUPRC, Specificity, BrierScore, TP, FN, FP, TN, PosAvg, NegAvg
            show = [
                epoch + 1, round(time, 2),round(ll, 5),round(accuracy, 5),round(rmse, 5),
                round(mae, 5),round(recall, 5),round(precision, 5),round(f1, 5),round(AUC, 5),
                round(AUPR, 5),round(specificity, 5),round(brier, 5),tp, fn_count, fp, tn,
                round(pos_avg, 5),round(neg_avg, 5)]
            print('\t\t'.join(map(str, show)))
            #================
             # 拼接成字符串
            # metrics_str = '\t\t'.join(map(str, show))
            metrics_str = (
                f"{epoch + 1:<6}"
                f"{time:<8.2f}"
                f"{ll:<10.5f}"
                f"{accuracy:<10.5f}"
                f"{rmse:<10.5f}"
                f"{mae:<10.5f}"
                f"{recall:<10.5f}"
                f"{precision:<10.5f}"
                f"{f1:<10.5f}"
                f"{AUC:<10.5f}"
                f"{AUPR:<10.5f}"
                f"{specificity:<10.5f}"
                f"{brier:<10.5f}"
                f"{tp:<6}"
                f"{fn_count:<6}"
                f"{fp:<6}"
                f"{tn:<6}"
                f"{pos_avg:<10.5f}"
                f"{neg_avg:<10.5f}"
            )
            # 关键：调用 log_result 写入文件
            log_msg(metrics_str)
            #================
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                # best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                # 记录最佳时刻的所有指标字符串，用于最后展示
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                # print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
                # 记录最佳AUC信息到文件（这也是“必要内容”）
                log_msg(f'AUC improved at epoch  {best_epoch} ;\tbest_auc: {best_auc}')
        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    # print('AUC:', AUCs)
    # AUC_mean = np.mean(AUCs)
    # AUC_std = np.std(AUCs)
    # print('Mean AUC:', AUC_mean, '(', AUC_std, ')')

    # print('AUPR:', AUPRs)
    # AUPR_mean = np.mean(AUPRs)
    # AUPR_std = np.std(AUPRs)
    # print('Mean AUPR:', AUPR_mean, '(', AUPR_std, ')')
        # === 【修改 5】: 记录最后的平均结果 ===
    log_msg(f'AUC: {AUCs}')
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    log_msg(f'Mean AUC: {AUC_mean} ( {AUC_std} )')

    log_msg(f'AUPR: {AUPRs}')
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    log_msg(f'Mean AUPR: {AUPR_mean} ( {AUPR_std} )')
    
    # 记得最后关闭文件
    f_log.close()
               # === Debugging Replacement Block ===
            


