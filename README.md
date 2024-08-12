# Marine-Predators-Algorithm
"""
Author: 王博民

This Python implementation of the Marine Predators Algorithm (MPA) is designed to solve both constrained and unconstrained
optimization problems. The algorithm is inspired by the strategies of marine predators in nature, focusing on the balance
between exploration and exploitation phases to effectively search the solution space.

The Marine Predators Algorithm is based on the following research: Faramarzi A, Heidarinejad M, Mirjalili S, et al. Marine Predators Algorithm:
A nature-inspired metaheuristic[J]. Expert systems with applications, 2020, 152: 113377.

Main Features: ------------------------------------------------ 
1. **Multi-phase Optimization**: - 
- **Phase 1** (HighVelocity Phase): Focuses on exploration where the search agents move rapidly to explore the solution space. 
- **Phase2** (Transition Phase): Balances exploration and exploitation where half of the agents continue exploring, and the other half begin exploiting the discovered good solutions.
- **Phase 3** (Low Velocity Phase): Focuses on exploitation, where the search agents exploit the best solutions found to refine them.

2. **Constraint Handling**: - The implementation can handle constrained optimization problems by incorporating a
repair function that modifies infeasible solutions to make them feasible.

3. **Early Stopping**: - An early stopping mechanism is included to terminate the optimization process when there is
no significant improvement in the best solution found, which helps to save computational resources.

4. **Adaptive Mechanisms**: - The algorithm uses mechanisms like Fish Aggregating Devices (FADs) to avoid local
optima and maintain diversity in the population.

5. **Progress Display**: - The algorithm includes a progress bar that provides real-time feedback on the optimization
process, including the current iteration, phase, and best fitness value.

Code Structure: ------------------------------------------------ 
- `stochastic_perturbation`: A function that
generates perturbations in the population to maintain diversity.
- `MarinePredatorsAlgorithm`: The main class
implementing the MPA, containing methods for initialization, fitness calculation, population update,
and visualization.
- `__initialize_population`: Initializes the population using a Latin Hypercube Sampling method.
-`__calculate_fitness`: Calculates the fitness of the population based on the provided fitness function.
-`__repair_func`: Repairs the population by making infeasible solutions feasible (if a repair function is provided).
-`optimizing`: The main method that performs the optimization process, iterating through the different phases and
applying the necessary updates to the population.
- `__visualize`: A method to visualize the fitness evolution over the iterations.

Applications:
------------------------------------------------
This algorithm can be applied to a wide range of optimization problems, including but not limited to:
- Engineering design optimization
- Machine learning model hyperparameter tuning
- Resource allocation problems
- Portfolio optimization in finance

The Marine Predators Algorithm is particularly effective in solving complex, multimodal optimization problems where
traditional optimization algorithms may struggle.

Usage:
------------------------------------------------
To use this implementation, create an instance of the `MarinePredatorsAlgorithm` class and call the `optimizing` method
with the appropriate parameters, including the fitness function, bounds, and population size. The algorithm will return
the best solution found and its corresponding fitness value.

"""
