import torch


def brownian_motion(current_position, only_step_size=False):
    next_position = current_position.clone()
    step_size = torch.randn(size=current_position.shape)
    if only_step_size:
        return step_size
    else:
        next_position += step_size
        return next_position


def show_brownian_motion():
    import matplotlib.pyplot as plt
    current_position = torch.rand(size=[1, 1])
    position = [current_position]
    for _ in range(1000):
        current_position = brownian_motion(current_position, only_step_size=True)
        position.append(current_position)
    position = torch.cat(position, dim=0)
    plt.figure()
    plt.title("Brownian Motion")
    plt.plot(position)
    plt.show()

    current_position = torch.rand(size=[4, 2])
    position = [current_position]
    for _ in range(100):
        current_position = brownian_motion(current_position)
        position.append(current_position)

    position = torch.stack(position)
    plt.figure()
    plt.title("Brownian Motion")
    plt.plot(position[:, :, 0], position[:, :, 1])
    plt.show()


if __name__ == '__main__':
    show_brownian_motion()
