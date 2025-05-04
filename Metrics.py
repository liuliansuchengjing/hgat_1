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
    def compute_effectiveness(self, yt_before, yt_after, inserted_lengths, topk_indices):
        """
        有效性计算（独立处理每个推荐资源）
        输入维度说明：
        yt_before: [B, seq_len, num_skills]（原始知识状态）
        yt_after: [B, seq_len-1, num_skills]（每个时间步插入后的最终知识状态）
        topk_indices: [B, seq_len-1, K]
        """
        batch_size, seq_len_minus_1, K = topk_indices.shape
        total_gain = 0.0
        valid_count = 0

        for b in range(batch_size):
            for t in range(seq_len_minus_1):
                recommended = topk_indices[b, t].tolist()
                valid_rec = [r for r in recommended if 0 <= r < yt_before.shape[2]]
                if not valid_rec:
                    continue

                # 原始知识状态（时间步t）
                pb_values = yt_before[b, t, valid_rec]  # [K_valid]
                # 插入后的知识状态（时间步t+K）
                pa_values = yt_after[b, t, valid_rec]  # [K_valid]

                for k in range(len(valid_rec)):
                    pb = pb_values[k].item()
                    pa = pa_values[k].item()
                    # if pb < 1.0 - 1e-6 and pa > 0:
                    if (pb < 1.0 - 1e-6) and (pa > 0):
                        gain = (pa - pb) / (1.0 - pb)
                        # print("pb:",pb)
                        # print("pa:",pa)
                        # print("----------")
                        total_gain += gain
                        valid_count += 1

        return total_gain / valid_count if valid_count > 0 else 0.0

    def gaintest_compute_metric(self, y_prob, y_true, batch_size, seq_len, k_list=[10, 50, 100], topnum=5):
        # 初始化所有指标字典，用于存储hits@k和map@k的累积值
        scores = {'hits@' + str(k): 0.0 for k in k_list}
        scores.update({'map@' + str(k): 0.0 for k in k_list})
        # 记录有效的样本数
        valid_samples = 0

        # 预初始化详细结果列表，确保长度与总样本数一致
        total_samples = batch_size * (seq_len - 1)
        detailed_results = [{'topk_resources': [], 'hits': 0.0} for _ in range(total_samples)]

        # 遍历每个样本
        for i in range(total_samples):
            # 获取当前样本的预测概率
            p_ = y_prob[i]  # 维度: [num_users]
            # 获取当前样本的真实标签
            y_ = y_true[i]
            # 如果真实标签为PAD，则跳过当前样本
            if y_ == self.PAD:
                continue  # 保持预初始化的空列表

            # 增加有效的样本数
            valid_samples += 1
            # 对预测概率进行排序
            p_sort = p_.argsort()
            # 获取TopK资源
            topk = p_sort[-topnum:][::-1] if topnum else p_sort[-5:][::-1]  # 维度: [topnum]

            # 直接更新详细结果的对应位置
            detailed_results[i]['topk_resources'] = topk.tolist()
            detailed_results[i]['hits'] = 1.0 if y_ in topk else 0.0

            # 计算hits@k和map@k
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

        # 生成TopK序列
        topk_sequence = []
        for b in range(batch_size):
            seq = []
            for t in range(seq_len - 1):
                idx = b * (seq_len - 1) + t
                if idx < len(detailed_results):
                    seq.append(detailed_results[idx]['topk_resources'][:topnum])
                else:
                    seq.append([])
            topk_sequence.append(seq)  # 维度: [batch_size, seq_len-1, topnum]

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
