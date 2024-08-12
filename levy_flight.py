import torch
import math


def calculator_sigma_x(alpha=1.5):
    sigma_x = math.gamma(1 + alpha) * math.sin(0.5 * math.pi * alpha)
    sigma_x = sigma_x / (math.gamma(0.5 * (1 + alpha)) * alpha * (2 ** (0.5 * alpha - 0.5)))
    sigma_x = sigma_x ** (1 / alpha)
    return sigma_x


def levy_flight(current_position, only_step_size=False):
    sigma_x = 0.6965745025576967
    sigma_y = 1.0
    gaussian_x = torch.normal(mean=0, std=sigma_x, size=current_position.size())
    gaussian_y = torch.normal(mean=0, std=sigma_y, size=current_position.size())
    step_size = 0.05 * gaussian_x / torch.pow(torch.abs(gaussian_y), 1 / 1.5)
    if only_step_size:
        return step_size
    else:
        return current_position + step_size


def show_levy_flight():
    import matplotlib.pyplot as plt
    current_position = torch.rand(size=[4, 2])
    position = [current_position]
    for _ in range(100):
        current_position = levy_flight(current_position)
        position.append(current_position)

    position = torch.stack(position)
    plt.figure()
    plt.title("Levy flight")
    plt.plot(position[:, :, 0], position[:, :, 1])
    plt.show()


if __name__ == '__main__':
    show_levy_flight()
