import torch


def reflective_boundaries(population: torch.Tensor,
                          lower_bound: list, upper_bound: list,
                          probability_reflective: float = 0.000001,
                          epsilon: float = 1e-6, device='cpu') -> torch.Tensor:
    """
    执行反射边界处理，将种群中超出边界的个体拉回到合法范围内。

    参数：
    - population (torch.Tensor): 种群矩阵，形状为 (size, dim)，其中 size 是个体数，dim 是维度数。
    - lower_bound (list): 每个维度的下界，长度应为 dim。
    - upper_bound (list): 每个维度的上界，长度应为 dim。
    - probability_reflective (float): 回弹概率，表示超出边界时，个体反射回边界的概率。默认值为 0.5。
    - epsilon (float): 一个很小的随机扰动量，用于避免数值不稳定性。默认值为 1e-6。
    - device (str): 计算设备，默认在 CPU 上执行。可以选择 "cuda" 来使用 GPU。

    返回：
    - torch.Tensor: 经过边界处理后的种群矩阵，形状与输入的 population 相同。

    功能描述：
    此函数的目的是处理种群中超出指定上下边界的个体。对于每个超出边界的个体，根据 `probability_reflective`
    的值，决定是否让该个体反射回边界（反射回边界的概率为 `probability_reflective`），或者根据某种规则
    拉回到合法范围内。如果超出下界，使用下界反射回或移动到边界内。如果超出上界，使用上界反射回或移动到
    边界内。

    处理逻辑：
    - 对每个维度，首先检查是否有个体超出下界或上界。
    - 对于超出下界的个体：
        - 如果随机数小于 `probability_reflective`，则个体反射回下界。
        - 否则，个体将拉回到合法范围内，并增加一个随机扰动量。
    - 对于超出上界的个体：
        - 如果随机数小于 `probability_reflective`，则个体反射回上界。
        - 否则，个体将拉回到合法范围内，并减少一个随机扰动量。
    - 如果所有个体都在合法范围内，则退出处理循环。

    """
    population = population.clone().to(device)
    size, dim = population.shape

    while True:
        all_in_bounds = True

        for i in range(dim):
            lb_i = torch.tensor(lower_bound[i], device=device).expand(size)
            ub_i = torch.tensor(upper_bound[i], device=device).expand(size)

            # 处理下界
            indices_low = population[:, i] < lb_i
            if indices_low.any():  # 检查是否有超出下界的个体
                all_in_bounds = False
                random_numbers_low = torch.rand(indices_low.sum().item(), device=device)
                population[indices_low, i] = torch.where(
                    random_numbers_low < probability_reflective,
                    lb_i[indices_low],
                    lb_i[indices_low] + (
                            lb_i[indices_low] - population[indices_low, i]) + epsilon * torch.randn(
                        indices_low.sum().item(), device=device)
                )

            # 处理上界
            indices_high = population[:, i] > ub_i
            if indices_high.any():  # 检查是否有超出上界的个体
                all_in_bounds = False
                random_numbers_high = torch.rand(indices_high.sum().item(), device=device)
                population[indices_high, i] = torch.where(
                    random_numbers_high < probability_reflective,
                    ub_i[indices_high],
                    ub_i[indices_high] - (
                            population[indices_high, i] - ub_i[indices_high]) - epsilon * torch.randn(
                        indices_high.sum().item(), device=device)
                )

        # 如果所有个体都在范围内，退出循环
        if all_in_bounds:
            break

    return population
