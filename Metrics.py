import numpy as np
import json
import pickle
import csv
import random
from collections import defaultdict
import networkx as nx
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score
import Constants
import itertools


def load_idx2u():
    with open('/kaggle/working/GCN/data/r_MOOC10000/idx2u.pickle', 'rb') as f:
        return pickle.load(f)


def load_u2idx():
    with open('/kaggle/working/GCN/data/r_MOOC10000/u2idx.pickle', 'rb') as f:
        return pickle.load(f)


def load_course_video():
    data = {}
    with open('/kaggle/input/riginmooccube/MOOCCube/relations/course-video.json', 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                course_id, video_id = row
                if course_id not in data:
                    data[course_id] = []
                data[course_id].append(video_id)
    return data


def load_course():
    courses = []
    with open('/kaggle/input/riginmooccube/MOOCCube/entities/course.json', 'r', encoding='utf-8') as f:
        data = f.read()
        start = 0
        end = 0
        while True:
            start = data.find('{', end)
            if start == - 1:
                break
            end = data.find('}', start) + 1
            json_obj = data[start:end]
            try:
                course = json.loads(json_obj)
                courses.append(course)
            except json.decoder.JSONDecodeError as e:
                print(f"解析错误: {e}")
    return courses


class Metrics(object):

    def __init__(self):
        super().__init__()
        self.PAD = 0

    def apk(self, actual, predicted, k=10):
        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        # if not actual:
        # 	return 0.0
        return score / min(len(actual), k)

    def compute_effectiveness(self, yt_before, yt_after, topk_indices):
        """
        精确计算每个推荐资源的有效性增益
        Args:
            yt_before: [batch_size, seq_len-1, num_skills]
            yt_after: [batch_size, seq_len-1, K, num_skills]
            topk_indices: [batch_size, seq_len-1, K]
        Returns:
            E_p: 平均有效性增益
        """
        batch_size, seq_len_minus_1, K = topk_indices.shape
        total_gain = 0.0
        valid_count = 0

        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                recommended = topk_indices[b, t].tolist()
                # 严格过滤无效索引（PAD和越界）
                valid_rec = [r for r in recommended
                             if r != Constants.PAD
                             and 0 <= r < yt_before.shape[2]]

                if not valid_rec:
                    continue

                # 精确提取对应位置的预测概率
                pb = yt_before[b, t, valid_rec]  # [num_valid]
                pa = yt_after[b, t, :len(valid_rec), valid_rec].diagonal()  # [num_valid]

                # 数值稳定性处理
                mask = (pb < 1.0 - 1e-6) & (pa >= 0) & (pa <= 1)  # 有效概率范围
                if not mask.any():
                    continue

                # 逐元素计算增益
                delta = (pa[mask] - pb[mask])
                denominator = (1.0 - pb[mask]).clamp(min=1e-6)  # 防止除零
                gain = (delta / denominator).sum().item()

                total_gain += gain
                valid_count += mask.sum().item()

        return total_gain / valid_count if valid_count > 0 else 0.0


    def compute_metric(self, y_prob, y_true, k_list=[10, 50, 100]):
        '''
            y_true: (#samples, )
            y_pred: (#samples, #users)
        '''
        scores_len = 0
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        scores = {'hits@' + str(k): [] for k in k_list}
        scores.update({'map@' + str(k): [] for k in k_list})
        for p_, y_ in zip(y_prob, y_true):
            if y_ != self.PAD:
                scores_len += 1.0
                p_sort = p_.argsort()
                for k in k_list:
                    topk = p_sort[-k:][::-1]
                    scores['hits@' + str(k)].extend([1. if y_ in topk else 0.])
                    scores['map@' + str(k)].extend([self.apk([y_], topk, k)])

        scores = {k: np.mean(v) for k, v in scores.items()}
        return scores, scores_len

    # Metrics.py 的 compute_effectiveness 方法
    # Metrics.py

    def compute_effectiveness(self, yt_before, yt_after, topk_indices):
        """
        计算推荐资源的有效性指标E_p
        Args:
            yt_before (Tensor): 原始知识状态 [batch_size, seq_len-1, num_skills]
            yt_after (Tensor): 插入后的知识状态 [batch_size, seq_len-1, K, num_skills]
            topk_indices (Tensor): TopK推荐索引 [batch_size, seq_len-1, K]
        Returns:
            E_p (float): 平均有效性增益
        """
        batch_size, seq_len_minus_1, K = topk_indices.shape
        total_gain = 0.0
        valid_count = 0

        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                # 获取当前时间步的推荐索引
                recommended = topk_indices[b, t].tolist()
                # 过滤无效索引（PAD和越界）
                valid_rec = [r for r in recommended if r != Constants.PAD and r < yt_before.shape[2]]
                if not valid_rec:
                    continue

                # 提取前后概率
                pb = yt_before[b, t, valid_rec]  # [num_valid]
                pa = yt_after[b, t, :len(valid_rec), valid_rec].diagonal()  # [num_valid]

                # 计算增益
                mask = (pb < 1.0 - 1e-6)  # 避免除零
                gain = ((pa[mask] - pb[mask]) / (1.0 - pb[mask])).sum()
                total_gain += gain.item()
                valid_count += mask.sum().item()

        return total_gain / valid_count if valid_count > 0 else 0.0

    def gaintest_compute_metric(self, y_prob, y_true, batch_size, seq_len, k_list=[10, 50, 100], topnum=None):
        # 初始化所有指标
        scores = {'hits@' + str(k): 0.0 for k in k_list}
        scores.update({'map@' + str(k): 0.0 for k in k_list})
        valid_samples = 0

        # 预初始化 detailed_results，确保长度与 total_samples 一致
        total_samples = batch_size * (seq_len - 1)
        detailed_results = [{'topk_resources': [], 'hits': 0.0} for _ in range(total_samples)]

        for i in range(total_samples):
            p_ = y_prob[i]
            y_ = y_true[i]
            if y_ == self.PAD:
                continue  # 保持预初始化的空列表

            valid_samples += 1
            p_sort = p_.argsort()
            topk = p_sort[-topnum:][::-1] if topnum else p_sort[-5:][::-1]

            # 直接更新 detailed_results 的对应位置
            detailed_results[i]['topk_resources'] = topk.tolist()
            detailed_results[i]['hits'] = 1.0 if y_ in topk else 0.0

            # 计算 hits@k 和 map@k
            for k in k_list:
                hits = 1.0 if y_ in p_sort[-k:][::-1] else 0.0
                scores['hits@' + str(k)] += hits
                scores['map@' + str(k)] += self.apk([y_], p_sort[-k:][::-1], k)

        # 归一化为均值
        for k in k_list:
            if valid_samples > 0:
                scores['hits@' + str(k)] /= valid_samples
                scores['map@' + str(k)] /= valid_samples
            else:
                scores['hits@' + str(k)] = 0.0
                scores['map@' + str(k)] = 0.0

        # 生成 topk_sequence
        topk_sequence = []
        for b in range(batch_size):
            seq = []
            for t in range(seq_len - 1):
                idx = b * (seq_len - 1) + t
                if idx < len(detailed_results):
                    seq.append(detailed_results[idx]['topk_resources'][:topnum])
                else:
                    seq.append([])
            topk_sequence.append(seq)

        return scores, topk_sequence, valid_samples


# Calculate accuracy of prediction result and its corresponding label
# output: tensor, labels: tensor
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)


class KTLoss(nn.Module):
    def __init__(self):
        super(KTLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='none')  # 不自动求平均

    def forward(self, pred_answers, real_answers, kt_mask):
        real_answers = real_answers[:, 1:].float()  # 截取有效部分并转为浮点
        answer_mask = kt_mask.bool()

        # --- 计算 AUC 和 ACC ---
        try:
            y_true = real_answers[answer_mask].cpu().detach().numpy()
            y_pred = pred_answers[answer_mask].cpu().detach().numpy()
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))  # 直接根据概率阈值计算ACC
        except ValueError:
            auc, acc = -1, -1

        # --- 计算带掩码的 BCE 损失 ---
        loss_per_element = self.bce_loss(pred_answers, real_answers)
        valid_loss = loss_per_element * answer_mask.float()  # 应用掩码
        loss = valid_loss.sum() / answer_mask.float().sum()  # 仅对有效位置求平均

        return loss, auc, acc

########
