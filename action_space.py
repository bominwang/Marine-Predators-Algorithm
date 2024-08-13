import torch
from brownian_motion import brownian_motion
from levy_flight import levy_flight


def high_velocity_action(prey_matrix: torch.Tensor, elite_matrix: torch.Tensor, P: float = 0.2, device=None):
    prey_matrix_clone = prey_matrix.clone()
    b_motion = brownian_motion(current_position=prey_matrix).to(device=device)
    step_size = b_motion * (elite_matrix - b_motion * prey_matrix)
    # U = torch.rand(size=[1, prey_matrix.shape[1]]).repeat(elite_matrix.shape[0], 1).to(device=device)
    U = torch.rand_like(elite_matrix).to(device=device)
    return prey_matrix_clone + P * U * step_size


def transition_action(prey_matrix: torch.Tensor, elite_matrix: torch.Tensor, P: float = 0.2, CF: float = None,
                      device=None):
    population_size = prey_matrix.shape[0]
    sliced_elite_matrix = elite_matrix[:int(0.5 * population_size), :]
    sliced_population_1 = prey_matrix[:int(0.5 * population_size), :]
    sliced_population_2 = prey_matrix[int(0.5 * population_size):, :]
    # exploration
    l_motion = levy_flight(current_position=sliced_population_1).to(device=device)
    step_size = l_motion * (sliced_elite_matrix - l_motion * sliced_population_1)
    # U = torch.rand(size=sliced_population_1.size()).to(device=device)
    U = torch.rand_like(sliced_population_1).to(device=device)
    sliced_population_1 = sliced_population_1 + P * U * step_size
    # exploitation
    b_motion = brownian_motion(current_position=sliced_population_2).to(device=device)
    step_size = b_motion * (b_motion * sliced_elite_matrix - sliced_population_2)
    sliced_population_2 = sliced_elite_matrix + P * CF * step_size
    return torch.cat([sliced_population_1, sliced_population_2], dim=0)


def low_velocity_action(prey_matrix: torch.Tensor, elite_matrix: torch.Tensor, P: float = 0.2, CF: float = None,
                        device=None):
    prey_matrix_clone = prey_matrix.clone()
    elite_matrix_clone = elite_matrix.clone()
    l_motion = levy_flight(current_position=prey_matrix_clone).to(device=device)
    step_size = l_motion * (l_motion * elite_matrix_clone - prey_matrix_clone)
    return elite_matrix_clone + P * CF * step_size

