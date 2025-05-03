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


def simulate_learning(kt_model, original_seqs, topk_sequence, graph, yt_before, batch_size, K):
    """
    模拟将推荐资源插入原始序列后的知识状态变化
    Args:
        kt_model: KTOnlyModel实例，仅含GNN和KT模块
        original_seqs: list[list[int]], 原始输入序列 [batch_size, seq_len]
        topk_sequence: list[list[list[int]]], 每个时间步的TopK推荐 [batch_size, seq_len-1, K]
        graph: Data, 预加载的图数据
        yt_before: Tensor, 原始知识状态 [batch_size, seq_len-1, num_skills]
        batch_size: int, 批大小
        K: int, TopK值
    Returns:
        yt_after: Tensor, 插入推荐后的知识状态 [batch_size, seq_len-1, K, num_skills]
    """
    seq_len_minus_1 = len(topk_sequence[0])  # 序列长度-1
    num_skills = yt_before.shape[-1]
    yt_after = torch.zeros(batch_size, seq_len_minus_1, K, num_skills).cuda()

    for t in range(seq_len_minus_1):
        extended_inputs = []  # 存储扩展后的序列
        extended_ans = []  # 存储模拟的答案

        # --- 为每个样本构建扩展序列 ---
        for b in range(batch_size):
            # 原始序列处理
            original_seq = original_seqs[b]  # list[int], 当前样本的原始序列
            actual_len = sum(1 for x in original_seq if x != Constants.PAD)  # 计算有效长度
            original_seq = original_seq[:actual_len]  # 截断PAD部分，得到实际序列

            # 获取当前时间步的TopK推荐（过滤PAD）
            recommended = topk_sequence[b][t]  # list[int], 当前时间步的推荐
            valid_rec = [r for r in recommended if r != Constants.PAD][:K]  # 有效推荐

            # 确定插入位置（当前时间步t+1的位置）
            insert_pos = t + 1
            if insert_pos > len(original_seq):
                insert_pos = len(original_seq)  # 不超过实际长度

            # 构建新序列：原始序列前insert_pos个元素 + 推荐资源
            new_seq = original_seq[:insert_pos] + valid_rec
            extended_inputs.append(new_seq)

            # 生成模拟答案：假设推荐资源预测正确率>0.5则正确
            pred_probs = yt_before[b, t, valid_rec].detach()  # Tensor[K]
            sim_answers = (pred_probs > 0.5).float().tolist()  # list[float]
            new_ans = [0.0] * insert_pos + sim_answers  # 原始部分答案设为0
            extended_ans.append(new_ans)

        # --- 对齐序列长度并转换为Tensor ---
        max_len = max(len(seq) for seq in extended_inputs)  # 当前批次最大长度
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long).cuda()  # [B, max_len]
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float).cuda()  # [B, max_len]

        for b in range(batch_size):
            seq = extended_inputs[b]
            ans = extended_ans[b]
            padded_inputs[b, :len(seq)] = torch.LongTensor(seq)  # 填充输入序列
            padded_ans[b, :len(ans)] = torch.FloatTensor(ans)  # 填充答案序列

        # --- 重新预测知识状态 ---
        with torch.no_grad():
            yt_after_batch = kt_model(padded_inputs, padded_ans, graph)  # [B, max_len, num_skills]

        # --- 提取推荐资源对应的知识状态 ---
        for b in range(batch_size):
            seq = extended_inputs[b]
            valid_rec = [r for r in topk_sequence[b][t] if r != Constants.PAD][:K]  # 有效推荐
            start_pos = len(seq) - len(valid_rec)  # 推荐资源起始位置

            if start_pos >= 0 and len(valid_rec) > 0:
                # 提取插入后的知识状态 [len(valid_rec), num_skills]
                yt_after[b, t, :len(valid_rec)] = yt_after_batch[b, start_pos:start_pos + len(valid_rec)]

        # 显存释放
        del padded_inputs, padded_ans, yt_after_batch
        torch.cuda.empty_cache()

    return yt_after


def gain_test_epoch(model, kt_model, test_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20], topnum=5):
    """
    执行测试流程，计算准确性指标和有效性指标
    Args:
        model: MSHGAT模型实例
        kt_model: KTOnlyModel实例
        test_data: DataLoader, 测试数据集
        graph: Data, 关系图数据
        hypergraph_list: list, 超图数据
        kt_loss: KTLoss实例
        k_list: list[int], 需要计算的TopK指标
        topnum: int, 实际使用的TopK值
    Returns:
        scores: dict, 各TopK的HitRate和MAP
        auc_list: list[float], AUC值列表
        acc_list: list[float], ACC值列表
        E_p: float, 平均有效性增益
    """
    model.eval()
    metric = Metrics()  # 指标计算实例
    scores = {f'hits@{k}': 0.0 for k in k_list}
    scores.update({f'map@{k}': 0.0 for k in k_list})
    total_valid = 0  # 有效样本总数
    auc_list, acc_list = [], []
    total_gain = 0.0  # 累计增益
    valid_count = 0  # 有效批次计数

    with torch.no_grad():
        for batch in test_data:
            # --- 数据准备 ---
            tgt, tgt_t, tgt_idx, ans = [x.cuda() for x in batch]  # tgt: [B, seq_len]
            batch_size, seq_len = tgt.shape

            # --- 模型前向传播 ---
            pred, pred_res, kt_mask, yt_before = model(tgt, tgt_t, tgt_idx, ans, graph, hypergraph_list)
            # pred: [B*(seq_len-1), num_skills]
            # yt_before: [B, seq_len-1, num_skills]

            # --- 计算传统指标 ---
            y_gold = tgt[:, 1:].reshape(-1).cpu().numpy()  # [B*(seq_len-1)]
            y_pred = pred.detach().cpu().numpy()  # [B*(seq_len-1), num_skills]

            # 计算HitRate和MAP
            batch_scores, _ = metric.compute_metric(y_pred, y_gold, k_list)
            for k in k_list:
                scores[f'hits@{k}'] += batch_scores[f'hits@{k}'] * batch_size * (seq_len - 1)
                scores[f'map@{k}'] += batch_scores[f'map@{k}'] * batch_size * (seq_len - 1)
            total_valid += batch_size * (seq_len - 1)

            # --- 生成TopK序列 ---
            _, topk_seq, _ = metric.gaintest_compute_metric(y_pred, y_gold, batch_size, seq_len, k_list, topnum)
            # topk_seq: list[list[list[int]]], [B, seq_len-1, K]

            # --- 转换格式为Tensor ---
            topk_tensor = torch.full((batch_size, seq_len - 1, topnum), Constants.PAD, dtype=torch.long).cuda()
            for b in range(batch_size):
                for t in range(seq_len - 1):
                    recs = topk_seq[b][t][:topnum]
                    topk_tensor[b, t, :len(recs)] = torch.LongTensor(recs)

            # --- 模拟学习并计算E_p ---
            yt_after = simulate_learning(kt_model, tgt.tolist(), topk_seq, graph, yt_before, batch_size, topnum)
            batch_gain = metric.compute_effectiveness(yt_before, yt_after, topk_tensor)
            total_gain += batch_gain
            valid_count += 1

    # --- 指标归一化 ---
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



