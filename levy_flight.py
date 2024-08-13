import torch


def levy_flight(current_position):
    sigma_x = 0.6965745025576967
    sigma_y = 1.0
    gaussian_x = torch.normal(mean=0, std=sigma_x, size=current_position.size())
    gaussian_y = torch.normal(mean=0, std=sigma_y, size=current_position.size())
    step_size = 0.05 * gaussian_x / torch.pow(torch.abs(gaussian_y), 1 / 1.5)
    return step_size
