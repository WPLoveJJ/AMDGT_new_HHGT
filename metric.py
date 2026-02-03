import numpy as np
# from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
#                              precision_recall_curve, roc_curve, roc_auc_score)
from sklearn.metrics import (auc, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, roc_curve, roc_auc_score, log_loss, brier_score_loss,
                             mean_squared_error, mean_absolute_error, confusion_matrix)



def get_metric(y_true, y_pred, y_prob):

    # accuracy = accuracy_score(y_true, y_pred)
    # mcc = matthews_corrcoef(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    # recall = recall_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred)

    # fpr, tpr, _ = roc_curve(y_true, y_prob)
    # Auc = auc(fpr, tpr)

    # precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
    # Aupr = auc(recall1, precision1)

    # return Auc, Aupr, accuracy, precision, recall, f1, mcc
    
     # 1. 原有指标
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        Auc = auc(fpr, tpr)
    except ValueError:
        Auc = 0.0

    # AUPR
    try:
        precision1, recall1, _ = precision_recall_curve(y_true, y_prob)
        Aupr = auc(recall1, precision1)
    except ValueError:
        Aupr = 0.0

    # 2. 新增指标 (LL, RMSE, MAE, BrierScore)
    # Log Loss
    ll = log_loss(y_true, y_prob, labels=[0, 1])
    
    # Brier Score
    brier = brier_score_loss(y_true, y_prob)
    
    # RMSE & MAE
    rmse = np.sqrt(mean_squared_error(y_true, y_prob))
    mae = mean_absolute_error(y_true, y_prob)

    # 3. 混淆矩阵与特异性 (TP, FN, FP, TN, Specificity)
    # ravel() 将矩阵展平为一维
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 4. 概率均值 (PosAvg, NegAvg)
    # 真实标签为1的样本的平均预测概率
    pos_mask = (y_true == 1)
    pos_avg = np.mean(y_prob[pos_mask]) if np.any(pos_mask) else 0.0
    
    # 真实标签为0的样本的平均预测概率
    neg_mask = (y_true == 0)
    neg_avg = np.mean(y_prob[neg_mask]) if np.any(neg_mask) else 0.0

    # 返回所有指标的元组
    return (Auc, Aupr, accuracy, precision, recall, f1, mcc, 
            ll, rmse, mae, specificity, brier, 
            tp, fn, fp, tn, pos_avg, neg_avg)