"""
Reference:
########################################################################################################################
# Phase 1: high velocity phase (number of iteration < 1/3 * maximum iteration number)
# Phase 2: unit velocity phase (number of iteration < 2/3 * maximum iteration number)
# Phase 3: low velocity phase (number of iteration < 3/3 * maximum iteration number)
########################################################################################################################
"""
import sys
import torch
import random
from brownian_motion import brownian_motion
from levy_flight import levy_flight
from boundaries import reflective_boundaries
import matplotlib.pyplot as plt


def stochastic_perturbation(population: torch.Tensor, num: int) -> torch.Tensor:
    size, dim = population.shape
    perturbation = torch.empty(size=[num, dim], device=population.device)

    for i in range(num):
        # 随机选择两个不同的索引
        idx = random.sample(range(size), 2)
        perturbation[i] = population[idx[0]] - population[idx[1]]

    return perturbation


class MarinePredatorsAlgorithm(object):
    def __init__(self, device=None):

        if device is None:
            # 如果不指定，默认利用GPU执行计算
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.lower_bound = None  # 捕食者下界
        self.upper_bound = None  # 捕食者上界
        self.dimension = None  # 捕食者维度
        self.fitness_function = None  # 适应度函数
        self.repair_function = None
        self.population_size = None  # 种群数量
        ################################################################################################################
        self.top_predator = None  # 历史最佳捕食者
        self.top_predator_history = []  # 历史最佳捕食者变化过程
        self.prey_history = []  # 种群轨迹
        self.fitness_history = []  # 种群适应度变化过程
        self.top_fitness = None  # 历史最佳捕食者适应度
        self.top_fitness_history = []  # 历史最佳捕食者适应度变化过程
        self.top_fitness_iterations = []  # 最佳捕食者适应度变化索引
        self.best_fitness = []  # 各阶段种群最佳适应度
        self.mean_fitness = []  # 各阶段种群平均适应度
        ################################################################################################################
        self.iter = 0  # 当前迭代次数
        self.p = 0.5
        self.FAD = 0.2
        self.CF = 0
        ################################################################################################################

    def __initialize_population(self):

        def latin_hypercube(size: int, dimension: int):
            section = torch.linspace(start=0, end=1, steps=size + 1)
            lb = section[:size].reshape(-1, 1).repeat(1, dimension)
            ub = section[1:size + 1].reshape(-1, 1).repeat(1, dimension)
            U = (lb + ub) / 2
            U = torch.transpose(U, 0, 1).reshape(-1, 1)
            index = torch.tensor(
                [random.sample([i + j * size for i in range(size)], size) for j in range(dimension)]).reshape(-1)
            return torch.transpose(U[index].reshape(dimension, size), 0, 1)

        population = latin_hypercube(size=self.population_size, dimension=self.dimension).to(device=self.device)
        upper_bound = torch.tensor(self.upper_bound, device=self.device)
        upper_bound = upper_bound.repeat(self.population_size, 1)
        lower_bound = torch.tensor(self.lower_bound, device=self.device)
        lower_bound = lower_bound.repeat(self.population_size, 1)
        population = lower_bound + (upper_bound - lower_bound) * population
        return population

    def __calculate_fitness(self, population: torch.Tensor):
        return self.fitness_function(population).reshape(self.population_size, 1).to(device=self.device)

    def __repair_func(self, population: torch.Tensor):
        return self.repair_function(population).reshape(self.population_size, self.dimension).to(device=self.device)

    def __visualize(self):
        plt.figure(figsize=(14, 10))

        # 绘制 top_fitness 曲线
        plt.plot(self.top_fitness_iterations, self.top_fitness_history, label="Top Fitness",
                 color='#DC4959', linestyle='-', linewidth=3, marker='o', markersize=8,
                 markerfacecolor='white', markeredgewidth=2)

        # 计算 mean_fitness 和 best_fitness 的 markevery 参数
        total_iterations = len(self.mean_fitness)
        markevery = max(1, total_iterations // 10)

        # 绘制 mean_fitness 曲线，只显示 10 个标记
        plt.plot(self.mean_fitness, label="Mean Fitness", color='#49DCB4', linestyle='--', linewidth=3,
                 marker='s', markersize=8, markerfacecolor='white', markeredgewidth=2, markevery=markevery)

        # 绘制 best_fitness 曲线，只显示 10 个标记
        plt.plot(self.best_fitness, label="Best Fitness", color='#07AEE3', linestyle='-.', linewidth=3,
                 marker='D', markersize=8, markerfacecolor='white', markeredgewidth=2, markevery=markevery)

        # 设置标签和标题
        plt.xlabel("Iteration", fontsize=18, fontweight='bold')
        plt.ylabel("Fitness", fontsize=18, fontweight='bold')
        plt.title("Fitness Evolution Over Iterations", fontsize=20, fontweight='bold')

        # 设置图例
        plt.legend(fontsize=20, loc='best', frameon=True, shadow=True, fancybox=True)

        # 网格和背景
        plt.grid(True, linestyle='--', linewidth=1.5)
        plt.gca().set_facecolor('#f0f0f0')

        # 展示图表
        plt.show()

    def optimizing(self, fitness_function: callable = None, repair_function: callable = None,
                   dimension: int = None, lower_bound: list = None, upper_bound: list = None,
                   population_size: int = None,
                   max_iteration: int = None,
                   early_stop: bool = False, tolerance: float = 1e-10, n_no_improvement: int = 10,
                   probability_reflective: float = 0.5):

        """
        优化函数，用于执行海洋捕食者算法（MPA）以解决给定的优化问题。

        参数:
        - fitness_function (callable): 适应度函数，用于评估种群中个体的优劣。这个函数不包含惩罚项。
        - repair_function (callable): 将不满足约束的种群进行修正，用于处理约束优化问题。
        - dimension (int): 决策变量的维度（问题的变量数量）。
        - lower_bound (list): 每个变量的下界，以列表形式提供，长度应与维度相同。
        - upper_bound (list): 每个变量的上界，以列表形式提供，长度应与维度相同。
        - population_size (int): 种群大小，即算法中并行计算的解的数量。
        - max_iteration (int): 最大迭代次数，即算法将执行的最大迭代步数。
        - early_stop (bool): 是否启用早停机制。如果启用，算法将在指定条件下提前停止。
        - tolerance (float): 早停机制中的容忍度。当最佳适应度变化低于此阈值时，算法将考虑没有显著改进。
        - n_no_improvement (int): 早停机制中允许的无显著改进的最大次数。如果超过此次数，算法将切换到下一阶段或终止。

        返回:
        - top_predator (list): 最优解，即在优化过程中找到的最佳解向量。
        - top_fitness (float): 最优解的适应度值。
        """

        self.fitness_function = fitness_function
        self.repair_function = repair_function
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size

        # ##############################################################################################################
        # Initialization
        # ##############################################################################################################

        # prey matrix
        prey_matrix = self.__initialize_population()
        if self.repair_function is not None:
            prey_matrix = self.__repair_func(prey_matrix)
        # calculate fitness
        fitness = self.__calculate_fitness(prey_matrix)
        # elite matrix
        idx_predator = torch.argmin(fitness).item()

        self.top_fitness = fitness[idx_predator].item()
        self.top_predator = prey_matrix[idx_predator]
        predator = self.top_predator.reshape(1, self.dimension)
        elite_matrix = predator.repeat(self.population_size, 1).to(device=self.device)

        self.iter = 0
        self.p = 0.5

        # record
        self.best_fitness = [self.top_fitness]
        self.prey_history.append(prey_matrix.clone())
        self.fitness_history.append(fitness.clone())
        self.mean_fitness.append(fitness.mean().item())
        self.top_fitness_history = [self.top_fitness]
        self.top_predator_history = [self.top_predator]
        self.top_fitness_iterations.append(self.iter)

        # 初始化早停机制参数
        stage_no_improvement_counter = 0
        best_seen_fitness = self.top_fitness
        current_phase = 1

        while self.iter < max_iteration:
            # ##########################################################################################################
            # early stop in each phase
            # ##########################################################################################################

            if early_stop:
                if best_seen_fitness - self.top_fitness > tolerance:
                    stage_no_improvement_counter = 0  # 如果有显著改进，重置计数器
                    best_seen_fitness = self.top_fitness  # 更新最佳适应度值
                else:
                    stage_no_improvement_counter += 1  # 否则，计数器加1

                # 判断是否需要切换阶段：基于早停条件或者基于迭代进度
                if stage_no_improvement_counter > n_no_improvement or (
                        self.iter >= max_iteration / 3 and current_phase == 1) or (
                        self.iter >= 2 * max_iteration / 3 and current_phase == 2):
                    stage_no_improvement_counter = 0  # 重置计数器

                    if current_phase == 1:
                        current_phase = 2
                    elif current_phase == 2:
                        current_phase = 3
                    elif current_phase == 3:
                        print(f"Completed all phases at iteration {self.iter}.")
                        break

            else:  # 如果没有启用早停机制，仅基于迭代进度切换阶段
                if self.iter >= max_iteration / 3 and current_phase == 1:
                    current_phase = 2
                elif self.iter >= 2 * max_iteration / 3 and current_phase == 2:
                    current_phase = 3

            # ##########################################################################################################
            # Detecting top predator
            # ##########################################################################################################

            if self.iter != 0:

                prey_matrix = reflective_boundaries(prey_matrix, self.lower_bound, self.upper_bound,
                                                    probability_reflective=probability_reflective).to(
                    device=self.device)

                if self.repair_function is not None:
                    prey_matrix = self.repair_function(prey_matrix)

                fitness = self.__calculate_fitness(prey_matrix).reshape(self.population_size, 1).to(device=self.device)
                idx_predator = torch.argmin(fitness).item()
                new_fitness = fitness[idx_predator].item()

                self.best_fitness.append(new_fitness)
                mean_fit = fitness.mean()
                self.mean_fitness.append(mean_fit.item())

                if new_fitness < self.top_fitness:
                    self.top_fitness = new_fitness
                    self.top_predator = prey_matrix[idx_predator]
                    elite_matrix = prey_matrix[idx_predator].reshape(1, self.dimension).to(device=self.device)

                    self.top_predator_history.append(self.top_predator)
                    self.top_fitness_history.append(self.top_fitness)
                    self.top_fitness_iterations.append(self.iter)

                # 计算进度和阶段信息
                progress = (self.iter + 1) / max_iteration * 100
                phase_name = ["High Velocity Phase", "Transition Phase", "Exploitation Phase"]
                progress_bar_length = 40  # 进度条的长度

                # 创建进度条
                progress_bar = f"[{'#' * int(progress / 100 * progress_bar_length)}" \
                               f"{'.' * (progress_bar_length - int(progress / 100 * progress_bar_length))}]"

                # 组合进度条显示内容
                status = f'Iteration: {self.iter}/{max_iteration}, {progress_bar} {progress:.2f}% Complete, ' \
                         f'Phase: {phase_name[current_phase - 1]}, Best Fitness: {self.top_fitness:.4f}'

                # 输出进度条并刷新
                sys.stdout.write('\r' + status)
                sys.stdout.flush()

                # 当算法执行完时，添加换行符避免覆盖
                if self.iter + 1 == max_iteration:
                    sys.stdout.write('\n')

                ########################################################################################################
                # Marine Memory Saving
                ########################################################################################################
                # 先获取当前和历史适应度及对应猎物的位置
                old_prey_matrix = self.prey_history[-1]
                old_fitness = self.fitness_history[-1]

                # 将当前适应度和历史适应度连接起来并排序
                combined_fitness = torch.cat((old_fitness, fitness), dim=0)
                sorted_fitness, sorted_indices = torch.sort(combined_fitness, dim=0)
                # 排序后的猎物矩阵（包括历史和当前的）
                combined_prey_matrix = torch.cat((old_prey_matrix, prey_matrix), dim=0)
                sorted_prey_matrix = combined_prey_matrix[sorted_indices.squeeze()]
                # 更新前一半的猎物矩阵和适应度值为最好的解
                prey_matrix = sorted_prey_matrix[:self.population_size]
                fitness = sorted_fitness[:self.population_size]

                # 保存更新后的猎物矩阵和适应度历史
                self.prey_history.append(prey_matrix.clone())
                self.fitness_history.append(fitness.clone())

            # ##########################################################################################################
            # update prey matrix based on phase
            # ##########################################################################################################

            # Phase 1 : High Velocity Phase (Exploration Phase)

            if current_phase == 1:
                b_motion = brownian_motion(current_position=prey_matrix, only_step_size=True).to(device=self.device)
                b_motion, elite_matrix, prey_matrix = b_motion.to(self.device), elite_matrix.to(
                    device=self.device), prey_matrix.to(device=self.device)
                step_size = b_motion * (elite_matrix - b_motion * prey_matrix).to(self.device)
                u = torch.rand(size=[1, self.dimension]).repeat(self.population_size, 1).to(device=self.device)
                prey_matrix = prey_matrix + self.p * u * step_size

            # Phase 2 : Transition Phase (Exploration + Exploitation)
            elif current_phase == 2:

                sliced_elite_matrix = elite_matrix[:int(0.5 * self.population_size), :].to(device=self.device)
                sliced_population_1 = prey_matrix[:int(0.5 * self.population_size), :].to(device=self.device)
                sliced_population_2 = prey_matrix[int(0.5 * self.population_size):, :].to(device=self.device)
                # Exploration
                l_motion = levy_flight(current_position=sliced_population_1, only_step_size=True).to(device=self.device)
                step_size = l_motion * (sliced_elite_matrix - l_motion * sliced_population_1)
                u = torch.rand(size=sliced_population_1.size()).to(device=self.device)
                sliced_population_1 = sliced_population_1 + self.p * u * step_size
                # exploitation
                self.CF = (1 - (self.iter / max_iteration)) ** (2 * (self.iter / max_iteration))
                b_motion = brownian_motion(current_position=sliced_population_2, only_step_size=True).to(
                    device=self.device)
                step_size = b_motion * (b_motion * sliced_elite_matrix - sliced_population_2)
                sliced_population_2 = sliced_elite_matrix + self.p * self.CF * step_size
                prey_matrix = torch.cat((sliced_population_1, sliced_population_2), dim=0)

            # phase 3: Exploitation Phase
            elif current_phase == 3:
                self.CF = (1 - (self.iter / max_iteration)) ** (2 * (self.iter / max_iteration))
                l_motion = levy_flight(current_position=prey_matrix, only_step_size=True).to(device=self.device)
                l_motion, elite_matrix, prey_matrix = l_motion.to(self.device), elite_matrix.to(
                    device=self.device), prey_matrix.to(device=self.device)
                step_size = l_motion * (l_motion * elite_matrix - prey_matrix)
                prey_matrix = elite_matrix + self.p * self.CF * step_size

            self.iter += 1

            # ##############################################################################################################
            # fish aggregating devices
            # ##############################################################################################################

            FADS = self.FAD * torch.ones(size=[self.population_size, 1]).to(device=self.device)
            RN = torch.rand(size=[self.population_size, 1]).to(device=self.device)
            prey_matrix_ = prey_matrix.clone()
            idx = RN < FADS
            idx = idx.reshape(-1)
            #
            sample_prey = prey_matrix[idx]
            self.CF = (1 - (self.iter / max_iteration)) ** (2 * (self.iter / max_iteration))
            population_min = torch.tensor(lower_bound).reshape(1, dimension).repeat(sample_prey.shape[0], 1).to(
                device=self.device)
            population_max = torch.tensor(upper_bound).reshape(1, dimension).repeat(sample_prey.shape[0], 1).to(
                device=self.device)
            random_matrix = torch.rand_like(sample_prey).to(self.device)
            U = (torch.rand(sample_prey.shape[0]) > 0.5).float().reshape(-1, 1).repeat(1, dimension).to(
                device=self.device)
            step_size = self.CF * (population_min + random_matrix * (population_max - population_min)) * U
            prey_matrix_[idx] = sample_prey + step_size
            #
            sample_prey = prey_matrix[~idx]
            random_matrix = torch.rand(sample_prey.shape[0], 1).repeat(1, dimension).to(device=self.device)
            step_size = self.FAD * (1 - random_matrix) + random_matrix * stochastic_perturbation(prey_matrix, num=int(
                population_size - idx.sum()))
            prey_matrix_[~idx] = sample_prey + step_size
            prey_matrix = prey_matrix_

        self.__visualize()

        return self.top_predator, self.top_fitness
