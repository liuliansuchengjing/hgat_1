import argparse
import time
import numpy as np
import Constants
import torch
import torch.nn as nn
from graphConstruct import ConRelationGraph, ConHyperGraphList
from dataLoader import Split_data, DataLoader
from Metrics import Metrics, KTLoss
from HGAT import MSHGAT, KTOnlyModel
from Optim import ScheduledOptim

metric = Metrics()

import torch
import numpy as np
from torch import nn
from Metrics import Metrics  # 保持原有Metrics类基础结构


def simulate_learning(kt_model, original_seqs, topk_sequence, graph, yt_before, batch_size, K):
    """
    模拟将推荐资源插入原始序列后的知识状态变化
    Args:
        kt_model (KTOnlyModel): 知识追踪模型，仅含GNN和KT模块
        original_seqs (list): 原始输入序列 [batch_size, seq_len]
        topk_sequence (list): 每个时间步的TopK推荐 [batch_size, seq_len-1, K]
        graph (Data): 预加载的图数据
        yt_before (Tensor): 原始知识状态 [batch_size, seq_len-1, num_skills]
        batch_size (int): 批大小
        K (int): TopK值
    Returns:
        yt_after (Tensor): 插入推荐后的知识状态 [batch_size, seq_len-1, K, num_skills]
    """
    seq_len_minus_1 = len(topk_sequence[0])  # seq_len-1
    yt_after = torch.zeros(batch_size, seq_len_minus_1, K, yt_before.shape[-1]).cuda()

    for t in range(seq_len_minus_1):
        # 为每个时间步构建扩展序列
        extended_inputs = []
        extended_ans = []
        for b in range(batch_size):
            original_seq = original_seqs[b]
            # 获取实际有效长度（去除PAD）
            actual_len = (torch.tensor(original_seq) != Constants.PAD).sum().item()
            original_seq = original_seq[:actual_len].tolist()

            # 获取当前时间步的TopK推荐（过滤PAD）
            recommended = topk_sequence[b][t]
            valid_rec = [r for r in recommended if r != Constants.PAD][:K]

            # 插入位置：当前时间步t+1（原始序列中t之后）
            insert_pos = t + 1
            if insert_pos > len(original_seq):
                insert_pos = len(original_seq)

            # 构建新序列：原始序列 + 推荐资源
            new_seq = original_seq[:insert_pos] + valid_rec
            extended_inputs.append(new_seq)

            # 生成模拟答案：假设推荐资源预测正确率>0.5则正确
            pred_probs = yt_before[b, t, valid_rec].detach()
            sim_answers = (pred_probs > 0.5).float().tolist()
            new_ans = [0.0] * insert_pos + sim_answers  # 原始部分答案设为0，推荐部分用预测值
            extended_ans.append(new_ans)

        # 对齐序列长度并转换为Tensor
        max_len = max(len(seq) for seq in extended_inputs)
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long).cuda()
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float).cuda()
        for b in range(batch_size):
            seq_len = len(extended_inputs[b])
            padded_inputs[b, :seq_len] = torch.LongTensor(extended_inputs[b]).cuda()
            padded_ans[b, :len(extended_ans[b])] = torch.FloatTensor(extended_ans[b]).cuda()

        # 重新预测知识状态
        with torch.no_grad():
            yt_after_batch = kt_model(padded_inputs, padded_ans, graph)  # [batch_size, max_len, num_skills]

        # 提取推荐资源对应的知识状态（插入后的最后K个时间步）
        for b in range(batch_size):
            seq_len = len(extended_inputs[b])
            valid_rec = [r for r in recommended if r != Constants.PAD][:K]  # 重新获取有效推荐

            # 严格校验插入位置
            start_pos = seq_len - len(valid_rec)
            if start_pos < 0 or len(valid_rec) == 0:
                continue

            # 确保提取范围有效
            end_pos = start_pos + len(valid_rec)
            if end_pos > yt_after_batch.shape[1]:
                continue

            # 精确写入目标位置
            yt_after[b, t, :len(valid_rec)] = yt_after_batch[b, start_pos:end_pos]

        # 释放显存
        del padded_inputs, padded_ans, yt_after_batch
        torch.cuda.empty_cache()

    return yt_after


def gain_test_epoch(model, kt_model, test_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20], topnum=5):
    """测试流程集成有效性计算"""
    model.eval()
    scores = {f'hits@{k}': 0.0 for k in k_list}
    scores.update({f'map@{k}': 0.0 for k in k_list})
    total_valid = 0
    auc_list, acc_list = [], []
    total_gain = 0.0
    valid_count = 0

    with torch.no_grad():
        for batch in test_data:
            # 数据准备
            tgt, tgt_t, tgt_idx, ans = [x.cuda() for x in batch]
            batch_size, seq_len = tgt.shape

            # 模型前向
            pred, pred_res, kt_mask, yt_before = model(tgt, tgt_t, tgt_idx, ans, graph, hypergraph_list)

            # 计算传统指标
            y_gold = tgt[:, 1:].reshape(-1).cpu()
            y_pred = pred.detach().cpu().numpy()
            batch_scores, _ = Metrics.compute_metric(y_pred, y_gold, k_list)
            for k in k_list:
                scores[f'hits@{k}'] += batch_scores[f'hits@{k}'] * batch_size * (seq_len - 1)
                scores[f'map@{k}'] += batch_scores[f'map@{k}'] * batch_size * (seq_len - 1)
            total_valid += batch_size * (seq_len - 1)

            # 生成TopK序列
            _, topk_seq, _ = Metrics.gaintest_compute_metric(y_pred, y_gold.numpy(), batch_size, seq_len, k_list, topnum)

            # 转换格式
            topk_tensor = torch.tensor(
                [[seq + [Constants.PAD] * (topnum - len(seq)) for seq in sample]
                 for sample in topk_seq],
                dtype=torch.long
            ).cuda()  # [batch, seq_len-1, K]

            # 模拟学习并计算E_p
            yt_after = simulate_learning(
                kt_model,
                tgt.tolist(),
                topk_seq,
                graph,
                yt_before,
                batch_size,
                topnum
            )
            batch_gain = Metrics.compute_effectiveness(yt_before, yt_after, topk_tensor)
            total_gain += batch_gain
            valid_count += 1

    # 归一化指标
    for k in k_list:
        scores[f'hits@{k}'] /= total_valid
        scores[f'map@{k}'] /= total_valid
    E_p = total_gain / valid_count if valid_count > 0 else 0.0

    return scores, auc_list, acc_list, E_p


def gain_test_model(model, data_path, opt):
    # ... 原有数据加载代码 ...
    kt_loss = KTLoss()
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    relation_graph = ConRelationGraph(data_path)
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    opt.user_size = user_size

    model = MSHGAT(opt, dropout=opt.dropout)
    model.load_state_dict(torch.load(opt.save_path))
    model.cuda()
    kt_loss = kt_loss.cuda()
    kt_model = KTOnlyModel(model)

    # 运行测试流程
    scores, auc_test, acc_test, E_p = gain_test_epoch(
        model, kt_model, test_data, relation_graph, hypergraph_list, kt_loss
    )

    # 打印结果
    print('\n===== 综合指标 =====')
    for metric in scores.keys():
        print(f'{metric}: {scores[metric]:.4f}')
    print(f'AUC: {np.mean(auc_test):.4f}')
    print(f'ACC: {np.mean(acc_test):.4f}')
    print(f'Effectiveness (E_p): {E_p:.4f}')

########



