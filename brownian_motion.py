import torch


def brownian_motion(current_position):
    step_size = torch.randn(size=current_position.shape)
    return step_size
