import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

import Constants
from dataLoader import Options
import pickle
import csv
import json


# 核心数据结构定义
class RecommendationProblem:
    def __init__(self, kt_model, yt_before, yt_after, original_seqs,original_ans,graph , topk_sequence, topk_indices, candidate_seq,
                 dataname, resource_embeddings, batch_size, seq_len, topnum, pred_probs, history_window=5):
        """
        :param yt_before: 原始知识状态 [batch_size, seq_len-1, num_skills]
        :param yt_after: 原始推荐后的知识状态 [batch_size, seq_len-1, num_skills]
        :param topk_sequence: 初始推荐序列 [batch_size, seq_len-1, K]
        :param candidate_matrix: 候选资源 [batch_size, seq_len-1, num_candidates]
        :param resource_difficulty: 资源难度字典 {resource_id: difficulty}
        :param resource_embeddings: 资源嵌入矩阵 [num_resources, embedding_dim]
        :param history_window: 禁止重复推荐的时间窗口
        """
        self.yt_before = yt_before
        self.yt_after = yt_after
        self.topk_sequence = topk_sequence
        self.topk_indices = topk_indices
        self.candidate_matrix = candidate_seq
        self.resource_difficulty = self.load_difficulty_dict(dataname)
        self.resource_embeddings = resource_embeddings.cpu().numpy()
        self.history_window = history_window
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.K = topnum
        self.similarity_matrix = 1 - cosine_similarity(self.resource_embeddings)
        self.kt_model = kt_model
        self.anslist = original_ans
        if not isinstance(original_ans, torch.Tensor):
            original_ans = torch.tensor(original_ans, dtype=torch.float)
        self.original_ans = original_ans
        self.original_seqs = original_seqs
        self.graph = graph
        self.pred_probs = torch.sigmoid(pred_probs).cpu().numpy() if pred_probs is not None else None

    def load_difficulty_dict(self, data_name):
        options = Options(data_name)

        # 加载索引到原始ID的映射
        with open(options.idx2u_dict, 'rb') as f:
            idx2u = pickle.load(f)  # 格式应为 [index: original_id] 的列表

        # 验证idx2u类型
        if not isinstance(idx2u, list):
            raise ValueError("idx2u应为列表格式，请检查数据生成过程")

        # 加载难度数据
        difficulty_data = {}

        # 根据文件格式处理
        if options.difficult_file.endswith('.csv'):
            with open(options.difficult_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        resource_id = int(row['challenge_id'])  # 列名需与实际文件匹配
                        difficulty = int(row['difficulty'])  # 直接转为整数
                        if difficulty not in {1, 2, 3}:
                            # logging.warning(f"非法难度值: {difficulty} (资源ID: {resource_id})")
                            continue
                        difficulty_data[resource_id] = difficulty
                    except (ValueError, KeyError) as e:
                        # logging.error(f"CSV数据解析失败: {e}")
                        continue

        elif options.difficult_file.endswith('.json'):
            with open(options.difficult_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    try:
                        if 'id' in item and 'diff' in item:
                            resource_id = int(item['id'])
                            difficulty = int(item['diff'])
                            if difficulty not in {1, 2, 3}:
                                # logging.warning(f"非法难度值: {difficulty} (资源ID: {resource_id})")
                                continue
                            difficulty_data[resource_id] = difficulty
                    except (ValueError, KeyError) as e:
                        # logging.error(f"JSON数据解析失败: {e}")
                        continue

        # 构建最终难度字典
        difficulty_dict = {}
        for idx, original_id in enumerate(idx2u):
            try:
                original_id = int(original_id)
            except ValueError:
                # logging.warning(f"无效的original_id: 索引={idx}, 值={original_id}")
                continue

            # 默认难度设为2（中等）
            difficulty_dict[idx] = difficulty_data.get(original_id, 1)

        return difficulty_dict


# 遗传算法核心类
class NSGA2Optimizer:
    def __init__(self, problem, population_size=50,
                 crossover_prob=0.9, mutation_prob=0.3):
        self.problem = problem
        self.pop_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

    def initialize_population(self, population_size):
        populations = {}
        for b in range(self.problem.batch_size):
            for t in range(self.problem.seq_len - 1):
                if self.problem.original_seqs[b][t] == Constants.PAD:
                    populations[(b, t)] = []
                    continue
                # 直接使用 Tensor 索引，并转换为列表
                original = self.problem.topk_indices[b, t].tolist()
                original = list(dict.fromkeys(original))  # 去重
                candidates = list(self.problem.candidate_matrix[b, t])
                candidates = [r for r in candidates if r not in original]
                population = []
                for _ in range(population_size):
                    individual = original[:self.problem.K // 3]
                    local_candidates = candidates.copy()
                    while len(individual) < self.problem.K and local_candidates:
                        chosen = np.random.choice(local_candidates)
                        individual.append(chosen)
                        local_candidates.remove(chosen)
                    if len(individual) < self.problem.K:
                        individual.extend(
                            [np.random.choice(candidates) for _ in range(self.problem.K - len(individual))])
                    individual = [int(item) for item in individual]
                    population.append(individual)
                populations[(b, t)] = population[:population_size]
        return populations

    def get_similarity_scores(self, anchor, candidates):
        """计算候选资源与锚资源的相似度"""
        anchor_embed = self.problem.resource_embeddings[anchor]
        candidate_embeds = self.problem.resource_embeddings[candidates]
        return np.exp(-0.5 * np.linalg.norm(candidate_embeds - anchor_embed, axis=1))

    # 目标函数计算
    def evaluate_individual(self, individual, batch_idx, time_step):
        # 输入检查
        assert len(individual) == self.problem.K, f"个体长度应为{self.problem.K}，实际{len(individual)}"

        """计算单个个体的三目标值"""
        # 1. 有效性计算（增量模拟）
        yt_after = self.simulate_learning(batch_idx, time_step, individual)
        effectiveness = self.calculate_effectiveness(batch_idx, time_step, individual, yt_after)

        # 2. 适应性计算
        adaptivity = self.calculate_adaptivity(batch_idx, time_step, individual)

        # 3. 多样性计算
        diversity = self.calculate_diversity(individual)

        # 4. 新增：准确率计算
        accuracy = self.calculate_accuracy(batch_idx, time_step, individual)

        assert len([effectiveness, adaptivity, diversity, accuracy]) == 4, "必须返回四个目标值"
        return [effectiveness, adaptivity, diversity, accuracy]

    def simulate_learning(self, batch_idx, time_step, recommended):
        """使用KT模型进行精确模拟"""
        # 获取当前知识状态
        current_state = self.problem.yt_before[batch_idx, time_step].clone()

        # 构建模拟输入序列
        input_seq = self.problem.original_seqs[batch_idx][:time_step + 1] + recommended
        # 使用 yt_before 模拟答题结果，大于0.5视为正确
        answer_seq = self.problem.anslist[batch_idx][:time_step + 1]  # 历史答案
        current_probs = self.problem.yt_before[batch_idx, time_step, recommended].cpu().numpy()  # 推荐资源的预测概率
        answer_seq += [1 if prob > 0.5 else 0 for prob in current_probs]  # 模拟推荐资源答案

        # 转换为张量
        input_tensor = torch.LongTensor(input_seq).unsqueeze(0).cuda()
        ans_tensor = torch.FloatTensor(answer_seq).unsqueeze(0).cuda()

        # 通过KT模型预测新状态
        with torch.no_grad():
            yt_all = self.problem.kt_model(input_tensor, ans_tensor, self.problem.graph)

        # 提取最终状态
        new_state = yt_all[0, -1, :].cpu().clone()
        return new_state

    def calculate_effectiveness(self, batch_idx, time_step, recommended, yt_after):
        """计算知识增益"""
        gain = 0.0
        for r in recommended:
            pb = self.problem.yt_before[batch_idx, time_step, r].item()
            pa = yt_after[r].item()
            epsilon = 1e-6
            denominator = max(1.0 - pb, epsilon)  # 确保分母至少为epsilon
            gain += max(0, pa - pb) / denominator
        return gain / len(recommended) if len(recommended) > 0 else 0.0

    def calculate_adaptivity(self, batch_idx, time_step, recommended):
        """计算适应性得分"""
        epsilon = 1e-6
        # 获取历史窗口内的资源和得分
        history_start = max(0, time_step - self.problem.history_window)
        # 使用 Tensor 索引并展平

        history_resources = self.problem.topk_indices[batch_idx, history_start:time_step].flatten().tolist()
        history_scores = self.problem.original_ans[batch_idx, history_start:time_step].flatten().tolist()
        diffs = [self.problem.resource_difficulty.get(r, 1) for r in history_resources]

        # 计算加权平均 delta
        weighted_sum = sum(d * s for d, s in zip(diffs, history_scores))
        score_sum = sum(history_scores) + epsilon
        delta = weighted_sum / score_sum if score_sum > epsilon else 1

        # 计算适应性得分
        adapt_sum = 0.0
        for r in recommended:
            diff_r = self.problem.resource_difficulty.get(r, 1.0)
            adapt_sum += 1 - abs(delta - diff_r)
        return adapt_sum / len(recommended) if len(recommended) > 0 else 0.0


    def calculate_diversity(self, recommended):
        """计算多样性得分"""
        if len(recommended) < 2:
            return 0.0
        embs = self.problem.resource_embeddings[recommended]
        sim_matrix = cosine_similarity(embs)
        return float(1 - sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean())

    def calculate_accuracy(self, batch_idx, time_step, recommended):
        """计算推荐资源的平均预测准确率"""
        if self.problem.pred_probs is None:
            return 0.0  # 若无预测数据，返回默认值

        # 获取当前batch和time_step对应的预测概率
        flat_index = batch_idx * (self.problem.seq_len - 1) + time_step
        resource_probs = self.problem.pred_probs[flat_index]  # 形状: [num_resources]

        # 计算推荐资源的平均概率
        acc_sum = 0.0
        for r in recommended:
            if r < len(resource_probs):
                acc_sum += resource_probs[r]
        return acc_sum / len(recommended) if len(recommended) > 0 else 0.0

    # 遗传操作实现
    def crossover(self, parent1, parent2, batch_idx, time_step):
        if np.random.rand() > self.crossover_prob:
            return parent1.copy()
        split = len(parent1) // 2
        child = parent1[:split]
        child += [r for r in parent2[split:] if r not in child]
        while len(child) < len(parent1):
            candidates = [r for r in self.problem.candidate_matrix[batch_idx, time_step] if r not in child]
            if not candidates:
                break
            child.append(np.random.choice(candidates))
        return child

    def mutate(self, individual, batch_idx, time_step):
        """变异操作：分层概率替换"""
        mutated = individual.copy()
        banned = set(individual)
        candidates = [r for r in self.problem.candidate_matrix[batch_idx, time_step]
                      if r not in banned]

        for i in range(len(mutated)):
            if i < 0.4 * len(mutated):  # 前40%低概率变异
                if np.random.rand() < 0.1 * (1 - i / len(mutated)):
                    if candidates:
                        mutated[i] = np.random.choice(candidates)
            else:  # 后60%正常变异
                if np.random.rand() < self.mutation_prob and candidates:
                    mutated[i] = np.random.choice(candidates)
        return mutated

    # NSGA-II核心流程
    def run(self, max_generations=100, convergence_thresh=0.05, population_size=10):
        """为每个 (batch, time_step) 运行独立优化"""
        all_populations = self.initialize_population(population_size)
        all_fronts = {}
        best_solutions = {}  # 存储每个 (b, t) 的最佳解
        weights = [0.3, 0.2, 0.2, 0.3]  # 对应 [ effectiveness, adaptivity, diversity, interest]

        for (b, t) in all_populations:
            if self.problem.original_seqs[b][t] == Constants.PAD:
                all_fronts[(b, t)] = []
                continue
            population = all_populations[(b, t)]
            front_history = []

            for gen in range(max_generations):
                # 评估种群
                fitness = [self.evaluate_individual(ind, b, t) for ind in population]

                # 非支配排序和选择
                selected_indices = self.nsga2_select(fitness)
                new_population = [population[i] for i in selected_indices]

                # 交叉和变异
                offspring = []
                while len(offspring) < population_size - len(new_population):
                    p1, p2 = np.random.choice(len(new_population), 2, replace=False)
                    child = self.crossover(new_population[p1], new_population[p2], b, t)
                    child = self.mutate(child, b, t)
                    offspring.append(child)

                population = new_population + offspring

                # 每10代检查收敛
                if gen % 3 == 0 and gen > 0:
                    current_front = self.get_pareto_front(population, b, t)
                    current_front_fitness = [self.evaluate_individual(ind, b, t) for ind in current_front]
                    if self.check_convergence(front_history, current_front_fitness, convergence_thresh):
                        # print(f"batch {b}, time_step {t} 在第{gen}代收敛!")
                        break
                    front_history.append(current_front_fitness)

            pareto_front = self.get_pareto_front(population, b, t)
            # 存储当前 (batch, time_step) 的 Pareto 前沿
            all_fronts[(b, t)] = pareto_front
            # 从 Pareto 前沿中选择指标和最大的个体
            if pareto_front:
                pareto_fitness = [self.evaluate_individual(ind, b, t) for ind in pareto_front]
                # 计算每个个体的指标和
                weights_array = np.array(weights)
                fitness_sums = np.sum(pareto_fitness * weights_array, axis=1)
                # 找到指标和最大的个体
                best_idx = np.argmax(fitness_sums)
                best_individual = pareto_front[best_idx]
                best_fitness = pareto_fitness[best_idx]
                best_sum = fitness_sums[best_idx]
                best_solutions[(b, t)] = (best_individual, best_fitness, best_sum)
            else:
                best_solutions[(b, t)] = (None, None, 0.0)

        # 返回所有 (batch, time_step) 的 Pareto 前沿
        return best_solutions

    def check_convergence(self, history, current_front_fitness, threshold):
        """检查 Pareto 前沿的收敛性"""
        if len(history) < 1:  # 需要至少一个历史前沿进行比较
            return False

        # 历史前沿的最后一个适应度值
        prev_front_fitness = np.array(history[-1])
        # 当前前沿的适应度值
        curr_front_fitness = np.array(current_front_fitness)

        # 计算前沿均值的变化率
        diff = np.linalg.norm(curr_front_fitness.mean(axis=0) - prev_front_fitness.mean(axis=0))
        norm_prev = np.linalg.norm(prev_front_fitness.mean(axis=0))

        return (diff / norm_prev) < threshold if norm_prev > 0 else False

    def nsga2_select(self, fitness):
        """完整的NSGA-II选择操作"""
        # 非支配排序
        fronts = self.fast_non_dominated_sort(fitness)

        if len(fronts) == 0:
            # 策略1: 随机选择个体作为新一代种群
            selected_indices = np.random.choice(len(fitness), self.pop_size, replace=False).tolist()
            return selected_indices

        # 计算拥挤度距离
        crowd_dist = self.crowding_distance(fitness, fronts)

        # 选择新一代种群
        selected = []
        front_idx = 0
        while (front_idx < len(fronts)) and (len(selected) + len(fronts[front_idx]) <= self.pop_size):
            selected += [(i, crowd_dist[i]) for i in fronts[front_idx]]
            front_idx += 1

        # 如果未填满，按拥挤度选择
        if len(selected) < self.pop_size:
            remaining = self.pop_size - len(selected)
            fronts[front_idx].sort(key=lambda x: crowd_dist[x], reverse=True)
            selected += [(i, crowd_dist[i]) for i in fronts[front_idx][:remaining]]

        return [i for i, _ in selected]

    def crowding_distance(self, fitness, fronts):
        """计算拥挤度距离"""
        crowd_dist = np.zeros(len(fitness))

        for front in fronts:
            # 按每个目标维度排序
            for m in range(4):  # 四个目标维度
                sorted_front = sorted(front, key=lambda x: fitness[x][m])

                # 边界个体距离设为无穷大
                crowd_dist[sorted_front[0]] = float('inf')
                crowd_dist[sorted_front[-1]] = float('inf')

                # 中间个体计算归一化距离
                f_min = fitness[sorted_front[0]][m]
                f_max = fitness[sorted_front[-1]][m]
                if f_max == f_min: continue  # 避免除以0

                for i in range(1, len(sorted_front) - 1):
                    crowd_dist[sorted_front[i]] += (
                    fitness[sorted_front[i + 1]][m] - fitness[sorted_front[i - 1]][m]) \
                                                   / (f_max - f_min)

        return crowd_dist

    def fast_non_dominated_sort(self, fitness):
        """快速非支配排序"""
        S = defaultdict(list)
        fronts = [[]]
        n = defaultdict(int)

        # 初始化支配关系
        for i in range(len(fitness)):
            for j in range(len(fitness)):
                if i == j: continue
                if self.dominates(fitness[i], fitness[j]):
                    S[i].append(j)
                elif self.dominates(fitness[j], fitness[i]):
                    n[i] += 1
            if n[i] == 0:
                fronts[0].append(i)

        # 若初始前沿为空（所有个体都被支配），直接返回空列表
        if not fronts[0]:
            return []

        # 构建后续前沿层
        i = 0
        while i < len(fronts):  # 动态检查前沿层索引
            current_front = fronts[i]
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            i += 1  # 移动到下一个前沿层
        return fronts

    def dominates(self, a, b):
        """判断个体a是否支配b"""
        # 三个目标都更优或相等，且至少有一个更优
        better = [x >= y for x, y in zip(a, b)]
        return all(better) and any([x > y for x, y in zip(a, b)])

    def get_pareto_front(self, population, batch_idx, time):
        """获取真实的Pareto前沿解"""
        fitness = [self.evaluate_individual(ind, batch_idx, time) for ind in population]
        pareto_front = []

        for i in range(len(population)):
            dominated = False
            for j in range(len(population)):
                if i == j: continue
                if self.dominates(fitness[j], fitness[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(population[i])

        return pareto_front

    def _batch_effectiveness(self, yt_after, recommendations):
        """向量化计算有效性"""
        pb = self.problem.yt_before[:, :-1, :].gather(2, recommendations.unsqueeze(-1))
        pa = yt_after.gather(2, recommendations.unsqueeze(-1))
        gains = (pa - pb) / (1.0 - pb + 1e-6)
        return gains.mean(dim=[1, 2])

    def _batch_adaptivity(self, recommendations):
        epsilon = 1e-6
        batch_size = self.problem.batch_size
        seq_len = self.problem.seq_len - 1
        history_window = self.problem.history_window

        # 扩展 recommendations 到 [batch_size, seq_len-1, K]
        recommendations_expanded = recommendations.view(batch_size, seq_len, -1)

        # 初始化 delta 数组 [batch_size, seq_len-1]
        delta = torch.zeros(batch_size, seq_len)

        # 计算每个时间步的 delta
        for b in range(batch_size):
            for t in range(seq_len):
                history_start = max(0, t - history_window)
                history_resources = self.problem.topk_sequence[b, history_start:t].flatten()
                history_scores = self.problem.original_ans[b, history_start:t].flatten()
                diffs = torch.tensor([self.problem.resource_difficulty.get(r, 1.0) for r in history_resources])
                weighted_sum = torch.sum(diffs * history_scores)
                score_sum = torch.sum(history_scores) + epsilon
                delta[b, t] = weighted_sum / score_sum if score_sum > epsilon else 1.0

        # 获取推荐资源的难度
        diff_r = torch.tensor([[self.problem.resource_difficulty.get(r, 1.0) for r in rec]
                               for rec in recommendations_expanded.view(-1, self.problem.K)]).view(batch_size, seq_len,
                                                                                                   self.problem.K)

        # 计算适应性
        adaptivity = 1 - torch.abs(delta.unsqueeze(-1) - diff_r)
        return adaptivity.mean(dim=[1, 2])  # 沿 batch 和时间步平均

    def _batch_diversity(self, recommendations):
        batch_size = self.problem.batch_size
        seq_len = self.problem.seq_len - 1
        K = self.problem.K

        # 扩展 recommendations 到 [batch_size, seq_len-1, K]
        recommendations_expanded = recommendations.view(batch_size, seq_len, K)

        # 获取嵌入向量
        embs = self.problem.resource_embeddings[recommendations_expanded.view(-1)]
        embs = embs.view(batch_size, seq_len, K, -1)

        # 计算相似度矩阵 [batch_size, seq_len, K, K]
        sim_matrix = cosine_similarity(embs.view(-1, embs.size(-1))).view(batch_size, seq_len, K, K)

        # 计算多样性 (1 - sim) 的均值，排除自相似度
        diversity_sum = torch.sum(
            (1 - sim_matrix) * (1 - torch.eye(K, K).to(sim_matrix.device)).unsqueeze(0).unsqueeze(0))
        num_pairs = batch_size * seq_len * (K * (K - 1) // 2)
        return diversity_sum / num_pairs if num_pairs > 0 else 0.0

    def _parallel_evaluate(self, batch):
        """GPU加速的批量评估"""
        # 将候选序列转为张量
        batch_tensor = torch.stack([torch.tensor(ind) for ind in batch]).cuda()

        # 批量模拟知识状态
        yt_after_batch = self.kt_model.batch_simulate(
            self.problem.yt_before,
            batch_tensor,
            self.problem.original_seqs,
            self.problem.original_ans
        )

        # 并行计算三个目标
        effectiveness = self.calculate_batch_effectiveness(yt_after_batch, batch_tensor)
        adaptivity = self.calculate_batch_adaptivity(batch_tensor)
        diversity = self.calculate_batch_diversity(batch_tensor)

        return list(zip(effectiveness, adaptivity, diversity))
