from customized_optimizer.CEC2022.cec_pytorch_packaging import pytorch_cec2022_func
from customized_optimizer.MPA.marine_predators import MarinePredatorsAlgorithm


def main(case, dim, max_iter, early_stop=False):
    if case == 1:
        # uni-modal function : shifted and full rotated zakharov function (*300)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 2:
        # uni-modal function : Shifted and full Rotated Rosenbrock’s Function (*400)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 3:
        # uni-modal function : Shifted and full Rotated Expanded Schaffer’s f6 Function (*600)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=1000,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 4:
        # uni-modal function : Shifted and full Rotated Non-Continuous Rastrigin’s Function (800)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=1000,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)
    if case == 5:
        # 支持维度[2, 10, 20]
        # uni-modal function : Shifted and full Rotated Levy Function (900)
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=1000,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)
    if case == 6:
        # uni-modal function : Hybrid Function (1800)
        # 支持维度[10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 7:
        # Hybrid Functions : Hybrid Function 2 (2000)
        # 支持维度[10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)
    if case == 8:
        # Hybrid Functions : Hybrid Function 3 (2200)
        # 支持维度[10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)
    if case == 9:
        # Composition Functions : Composition Function 1  (2300)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 10:
        # Composition Functions : Composition Function 2  (2400)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 11:
        # Composition Functions : Composition Function 3  (2600)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)

    if case == 12:
        # Composition Functions : Composition Function 4  (2700)
        # 支持维度[2, 10, 20]
        dim = dim
        lower_bound = [-100] * dim
        upper_bound = [100] * dim
        optimizer = MarinePredatorsAlgorithm()
        func = pytorch_cec2022_func(func_num=case)
        c, d = optimizer.optimizing(fitness_function=func,
                                    dimension=dim, lower_bound=lower_bound, upper_bound=upper_bound,
                                    population_size=50,
                                    max_iteration=max_iter,
                                    early_stop=early_stop, tolerance=1e-6, n_no_improvement=10)


if __name__ == '__main__':
    main(case=8, dim=10, max_iter=2000, early_stop=False)
