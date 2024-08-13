import torch
import random


def stochastic_perturbation(population: torch.Tensor, num: int) -> torch.Tensor:
    size, dim = population.shape
    perturbation = torch.empty(size=[num, dim], device=population.device)

    for i in range(num):
        # 随机选择两个不同的索引
        idx = random.sample(range(size), 2)
        perturbation[i] = population[idx[0]] - population[idx[1]]

    return perturbation


def fish_aggregating_device(prey_matrix: torch.Tensor, lower_bound: list, upper_bound: list, CF: float, FAD: float,
                            device=None):
    size, dim = prey_matrix.shape
    FADS = FAD * torch.ones(size=[size, 1]).to(device=device)
    RN = torch.rand(size=[size, 1]).to(device=device)
    prey_matrix_clone = prey_matrix.clone()

    idx = RN < FADS
    idx = idx.reshape(-1)
    # FADS > RN
    sample_prey = prey_matrix[idx]
    population_min = torch.tensor(lower_bound).reshape(1, dim).repeat(sample_prey.shape[0], 1).to(device=device)
    population_max = torch.tensor(upper_bound).reshape(1, dim).repeat(sample_prey.shape[0], 1).to(device=device)
    random_matrix = torch.rand_like(sample_prey).to(device=device)
    # U = (torch.rand(sample_prey.shape[0]) > 0.2).float().reshape(-1, 1).repeat(1, dim).to(device=device)
    U = torch.rand_like(sample_prey).to(device=device)
    step_size = CF * (population_min + random_matrix * (population_max - population_min)) * U
    prey_matrix_clone[idx] = sample_prey + step_size
    # FADS < RN
    sample_prey = prey_matrix[~idx]
    # random_matrix = torch.rand(sample_prey.shape[0], 1).repeat(1, dim).to(device=device)
    random_matrix = torch.rand_like(sample_prey).to(device=device)
    step_size = FAD * ((1 - random_matrix) + random_matrix) * stochastic_perturbation(population=prey_matrix,
                                                                                      num=int(prey_matrix.shape[0]
                                                                                              - idx.sum()))
    prey_matrix_clone[~idx] = sample_prey + step_size
    return prey_matrix_clone
