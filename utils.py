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

# 初始化一个Metrics类的实例，用于后续的指标计算
metric = Metrics()

import torch
import numpy as np
from torch import nn
from Metrics import Metrics  # 保持原有Metrics类基础结构

def simulate_learning(kt_model, original_seqs, original_ans, topk_sequence, graph, yt_before, batch_size, K):
    """
    单时间步模拟学习（推荐插入到当前时间步之后，仅保留到t+K）
    输入维度说明：
    original_seqs: list[list[int]] 形状 [batch_size, original_seq_len]
    original_ans: list[list[int]] 形状 [batch_size, original_seq_len]
    topk_sequence: list[list[list[int]] 形状 [batch_size, seq_len-1, K]
    yt_before: Tensor 形状 [batch_size, seq_len-1, num_skills]
    """


    seq_len = len(topk_sequence[0]) # 推荐时间步数（seq_len - 1）
    yt_after_list = []  # 存储每个时间步插入后的知识状态

    for t in range(seq_len):# 遍历每个推荐时间步
        extended_inputs = []# 扩展后的输入序列，形状 [batch_size, insert_pos + K]
        extended_ans = []# 扩展后的答案序列，形状 [batch_size, insert_pos + K]

        for b in range(batch_size):# 遍历每个样本
            original_seq = original_seqs[b]  # 获取当前样本的原始序列, 维度: [original_seq_len]
            original_an = original_ans[b]
            recommended = topk_sequence[b][t]  # 获取当前时间步的推荐资源, 维度: [K]
            insert_pos = t + 1  # 插入位置（当前时间步之后）
            new_seq = original_seq[:insert_pos] + recommended# 构建新的序列，包含原始序列前t+1个元素和推荐的K个元素

            extended_inputs.append(new_seq)# 将新序列添加到扩展输入列表中 [insert_pos + K]

            pred_answers = (yt_before[b, t, recommended] > 0.4).float().tolist() # 基于yt_before的当前时间步t生成预测答案
            new_ans = original_an[:insert_pos] + pred_answers # 构建答案序列，前t+1个位置为原始序列答案，推荐位置为预测答案
            extended_ans.append(new_ans)# 将新答案序列添加到扩展答案列表中# [insert_pos + K]

        # 确定扩展后序列的最大长度
        max_len = insert_pos + K
        # 初始化填充后的输入张量，使用PAD填充 [batch_size, max_len]
        padded_inputs = torch.full((batch_size, max_len), Constants.PAD, dtype=torch.long).cuda()  # 维度: [batch_size, max_len]
        # 初始化填充后的答案张量，初始值为0 # [batch_size, max_len]
        padded_ans = torch.zeros((batch_size, max_len), dtype=torch.float).cuda()  # 维度: [batch_size, max_len]

        # 遍历每个样本，将扩展后的序列和答案填充到张量中
        for b in range(batch_size):
            seq = extended_inputs[b]
            ans = extended_ans[b]
            padded_inputs[b, :len(seq)] = torch.LongTensor(seq).cuda()# 填充输入
            padded_ans[b, :len(ans)] = torch.FloatTensor(ans).cuda()# 填充答案

        # 重新预测知识状态，不进行梯度计算
        with torch.no_grad():
            yt_after = kt_model(padded_inputs, padded_ans, graph)  # 维度: [batch_size, max_len, num_skills]
        # 提取插入后的知识状态（时间步t+K，即扩展后的最后一个时间步），并移至CPU
        p_after = yt_after[:, -1, :].detach().cpu()  # 维度: [batch_size, num_skills]
        # 将插入后的知识状态添加到列表中
        yt_after_list.append(p_after)

        # 显式释放GPU内存
        del padded_inputs, padded_ans, yt_after
        torch.cuda.empty_cache()

    # 将所有时间步的插入后的知识状态堆叠成一个张量
    return torch.stack(yt_after_list, dim=1)  # 维度: [batch_size, seq_len-1, num_skills]


def gain_test_epoch(model, kt_model, test_data, graph, hypergraph_list, kt_loss, k_list=[5, 10, 20], topnum=1):
    model.eval()# 将模型设置为评估模式
    auc_test, acc_test = [], []
    scores = {'hits@' + str(k): 0.0 for k in k_list}
    scores.update({'map@' + str(k): 0.0 for k in k_list})
    total_valid_samples = 0 # 记录有效的样本总数
    total_gain = 0.0    # 有效性指标的累积值
    total_valid_count = 0     # 有效的批次数量

    with torch.no_grad():    # 不进行梯度计算
        for i, batch in enumerate(test_data):  # 遍历测试数据的每个批次
            tgt, tgt_timestamp, tgt_idx, ans = batch    # 解包批次数据
            batch_size, seq_len = tgt.size()  # tgt维度: [batch_size, seq_len]
            tgt = tgt.cuda()
            ans = ans.cuda()
            # 前向传播，得到预测结果、预测资源、掩码和原始知识状态
            pred, pred_res, kt_mask, yt_before = model(tgt, tgt_timestamp.cuda(), tgt_idx.cuda(), ans, graph, hypergraph_list)
            # pred维度: [batch_size*seq_len-1, num_users]
            # pred_res维度: [batch_size, seq_len]
            # kt_mask维度: [batch_size, seq_len]
            # yt_before维度: [batch_size, seq_len-1, num_skills]

            # # 将数据转移到CPU并转换为NumPy
            # tgt_np = tgt.cpu().numpy()  # [batch_size, seq_len]
            # ans_np = ans.cpu().numpy()  # [batch_size, seq_len]
            # yt_before_np = yt_before.cpu().numpy()  # [batch_size, seq_len-1, num_skills]
            #
            # batch_size, seq_len = tgt_np.shape
            #
            # # 遍历每个样本
            # for b in range(batch_size):
            #     print(f"\n=== Batch {i}, Sample {b} ===")
            #
            #     # 打印 tgt 前10个时间步（题目ID）
            #     print("[tgt] First 10 steps:", list(tgt_np[b, :10]))
            #
            #     # 打印 ans 前10个时间步（答题结果）
            #     print("[ans] First 10 steps:", list(ans_np[b, :10].round(2)))  # 保留两位小数
            #
            #     # 打印 yt_before 中对应 tgt 的每个时间步的概率值
            #     print("[yt_before] Corresponding skill probabilities:")
            #     for t in range(10):
            #         if t >= seq_len - 1:
            #             break  # 避免越界（yt_before只有seq_len-1个时间步）
            #
            #         for i in range(10):
            #             # 获取当前时间步 t+1 的题目ID（因为 yt_before[t] 预测的是 t+1 的题目）
            #             skill_id = tgt_np[b, i]
            #
            #             # 跳过PAD或非法ID
            #             if skill_id == Constants.PAD or skill_id >= yt_before_np.shape[2]:
            #                 continue
            #             prob = yt_before_np[b, t, skill_id]
            #             print(f"  Step {i}: SkillID={skill_id} -> Prob={prob:.4f}")


            loss_kt, auc, acc = kt_loss(pred_res, ans, kt_mask)# 计算当前批次的AUC和ACC
            if auc != -1:    # 如果AUC计算成功，则将其添加到列表中
                auc_test.append(auc)
                acc_test.append(acc)

            # 处理预测结果，生成TopK序列
            y_gold = tgt[:, 1:].contiguous().view(-1).cpu().numpy()  # 维度: [(batch_size * (seq_len - 1))]
            y_pred = pred.detach().cpu().numpy()  # 维度: [batch_size*seq_len-1, num_users]
            # print("y_pred",y_pred.shape)# 维度: [batch_size, seq_len-1, num_users]
            # print("y_gold",y_gold.shape) # 维度: [(batch_size * (seq_len - 1))]
            scores_batch, topk_sequence, scores_len = metric.gaintest_compute_metric(
                y_pred, y_gold, batch_size, seq_len, k_list, topnum
            )
            # 累加有效的样本数
            total_valid_samples += scores_len

            # 累加每个指标的值
            for k in k_list:
                scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
                scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len

            # 模拟学习（逐时间步插入）
            original_seqs = tgt.tolist()  # 维度: [batch_size, seq_len]
            original_ans = ans.tolist()  # 维度: [batch_size, seq_len]
            topk_indices = torch.tensor(
                [[rec + [Constants.PAD] * (topnum - len(rec)) for rec in sample] for sample in topk_sequence],
                dtype=torch.long
            ).cuda()  # 维度: [batch_size, seq_len-1, topnum]

            yt_after = simulate_learning(
                kt_model, original_seqs, original_ans, topk_sequence, graph, yt_before, batch_size, topnum
            )  # 维度: [batch_size, seq_len-1, num_skills]

            # 计算有效性（仅当前时间步）
            batch_gain = metric.compute_effectiveness(
                yt_before,  # [batch_size, seq_len-1, num_skills]
                yt_after,  # [batch_size, seq_len-1, num_skills]
                torch.zeros(batch_size, dtype=torch.long),  # 不需要插入长度累积 [batch_size]
                topk_indices  # [batch_size, seq_len-1, topnum]
            )
            # 累加有效性指标
            total_gain += batch_gain
            # 累加有效的批次数量
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
    # 初始化KTLoss类的实例
    kt_loss = KTLoss()
    # 分割数据，得到用户数量、总级联数、时间戳、训练集、验证集和测试集
    user_size, total_cascades, timestamps, train, valid, test = Split_data(data_path, opt.train_rate, opt.valid_rate,
                                                                           load_dict=True)

    # 创建测试数据加载器
    test_data = DataLoader(test, batch_size=opt.batch_size, load_dict=True, cuda=False)

    # 构建关系图
    relation_graph = ConRelationGraph(data_path)
    # 构建超图列表
    hypergraph_list = ConHyperGraphList(total_cascades, timestamps, user_size)

    # 将用户数量添加到选项中
    opt.user_size = user_size

    # 初始化MSHGAT模型
    model = MSHGAT(opt, dropout=opt.dropout)
    # 加载预训练模型参数
    model.load_state_dict(torch.load(opt.save_path))
    # 将模型移到GPU上
    model.cuda()
    # 将KTLoss移到GPU上
    kt_loss = kt_loss.cuda()
    # 初始化KTOnlyModel
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
