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
    单时间步模拟学习（推荐插入到当前时间步之后，仅保留到t+K）
    输入维度说明：
    original_seqs: list[list[int]] 形状 [batch_size, original_seq_len]
    topk_sequence: list[list[list[int]] 形状 [batch_size, seq_len-1, K]
    yt_before: Tensor 形状 [batch_size, seq_len, num_skills]
    """
    seq_len = len(topk_sequence[0])
    yt_after_list = []

    for t in range(seq_len):
        extended_inputs = []
        extended_ans = []
        for b in range(batch_size):
            # 获取原始序列并去除填充部分（PAD=0）
            original_seq = original_seqs[b]
            actual_len = (torch.tensor(original_seq) != Constants.PAD).sum().item()  # 实际有效长度
            original_seq = original_seq[:actual_len]  # 截断到有效部分

            recommended = topk_sequence[b][t]
            # 过滤推荐资源中的PAD标记
            valid_rec = [r for r in recommended if r != Constants.PAD]

            insert_pos = t + 1
            if insert_pos > len(original_seq):  # 插入位置不超过实际长度
                insert_pos = len(original_seq)

            new_seq = original_seq[:insert_pos] + valid_rec[:K]  # 仅保留前K个有效推荐
            extended_inputs.append(new_seq)

            # 生成预测答案（基于yt_before的当前时间步t）
            pred_answers = (yt_before[b, t, recommended] > 0.5).float().tolist()
            # 答案序列：前t+1个位置为原始答案（默认0），推荐位置为预测答案
            new_ans = [0.0] * (insert_pos) + pred_answers
            extended_ans.append(new_ans)

        # 转换为张量并填充到统一长度（t+1 + K）
        max_len = insert_pos + K  # 扩展后的最大长度（t+1 + K）
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long).cuda()
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float).cuda()

        for b in range(batch_size):
            seq = extended_inputs[b]
            ans = extended_ans[b]
            padded_inputs[b, :len(seq)] = torch.LongTensor(seq).cuda()
            padded_ans[b, :len(ans)] = torch.FloatTensor(ans).cuda()

        # 重新预测知识状态
        with torch.no_grad():
            yt_after = kt_model(
                padded_inputs,
                padded_ans,
                graph
            )
        # 提取插入后的知识状态（时间步t+K，即扩展后的最后一个时间步）
        # p_after = yt_after[:, -1, :]  # [B, num_skills]
        # yt_after_list.append(p_after)
        # 分离张量并转为CPU（若允许）
        p_after = yt_after[:, -1, :].detach().cpu()  # 分离计算图并移至CPU
        yt_after_list.append(p_after)

        # 显式释放GPU内存
        del padded_inputs, padded_ans, yt_after
        torch.cuda.empty_cache()

    return torch.stack(yt_after_list, dim=1)  # [B, seq_len-1, num_skills]



def gain_test_epoch(model, kt_model, test_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20], topnum=5):
    model.eval()
    auc_test, acc_test = [], []
    # 初始化指标
    scores = {'hits@' + str(k): 0.0 for k in k_list}
    scores.update({'map@' + str(k): 0.0 for k in k_list})
    total_valid_samples = 0
    # 有效性指标的累积
    total_gain = 0.0
    total_valid_count = 0

    with torch.no_grad():
        for i, batch in enumerate(test_data):
            tgt, tgt_timestamp, tgt_idx, ans = batch
            batch_size, seq_len = tgt.size()

            # 前向传播
            tgt = tgt.cuda()
            ans = ans.cuda()
            pred, pred_res, kt_mask, yt_before = model(tgt, tgt_timestamp.cuda(), tgt_idx.cuda(), ans, graph, hypergraph_list)

            # 计算当前批次的 AUC 和 ACC
            loss_kt, auc, acc = kt_loss(pred_res, ans, kt_mask)
            if auc != -1:
                auc_test.append(auc)
                acc_test.append(acc)

            # 处理预测结果 生成 TopK 序列
            y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            scores_batch, topk_sequence, scores_len = metric.gaintest_compute_metric(
                y_pred, y_gold, batch_size, seq_len, k_list, topnum
            )
            total_valid_samples += scores_len

            # 累加指标
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            # 模拟学习（逐时间步插入）
            original_seqs = tgt.tolist()
            topk_indices = torch.tensor(
                [[rec + [Constants.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
                dtype=torch.long
            ).cuda()  # [B, seq_len-1, topnum]

            yt_after = simulate_learning(
                kt_model, original_seqs, topk_sequence, graph, yt_before, batch_size, topnum
            )

            # 计算有效性（仅当前时间步）
            batch_gain = metric.compute_effectiveness(
                yt_before,  # 原始知识状态（排除最后一个时间步）
                yt_after,
                torch.zeros(batch_size, dtype=torch.long),  # 不需要插入长度累积
                topk_indices
            )
            total_gain += batch_gain
            total_valid_count += 1

    # 计算全局均值
    for k in k_list:
        if total_valid_samples > 0:
            scores['hits@' + str(k)] /= total_valid_samples
            scores['map@' + str(k)] /= total_valid_samples
        else:
            scores['hits@' + str(k)] = 0.0
            scores['map@' + str(k)] = 0.0

    # 计算平均有效性
    E_p = total_gain / total_valid_count if total_valid_count > 0 else 0.0

    return scores, auc_test, acc_test, E_p


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



