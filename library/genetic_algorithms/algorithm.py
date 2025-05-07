from typing import Callable
from copy import deepcopy
from solution import Solution
import random


def get_best_ind(population: list[Solution], maximization: bool):
    """
    Calculates the fitness of each individual in the population and 
    selects the individual with the highest or lowest fitness, depending on the 
    value of the 'maximization' argument.

    Args:
        population (list[Solution]): A list of Solution objects representing the population.
        maximization (bool): If True, the function selects the individual with the highest fitness (maximization).
                              If False, the function selects the individual with the lowest fitness (minimization).

    Returns:
        Solution: The individual with best fitness in the population.
    """

    fitness_list = [ind.fitness() for ind in population]

    if maximization:
        return population[fitness_list.index(max(fitness_list))]
    else:
        return population[fitness_list.index(min(fitness_list))]


def genetic_algorithm(
    initial_population: list[Solution],
    max_gen: int,
    selection_algorithm: Callable,
    maximization: bool = False,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: bool = True,
    verbose: bool = False,
):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """

    # Initialize a population with N individuals
    population = initial_population

    # Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose:
            print(f'-------------- Generation: {gen} --------------')

        # Create an empty population P'
        new_population = []

        # If using elitism, insert best individual from P into P'
        if elitism:
            best_ind= get_best_ind(population, maximization)
            new_population.append(deepcopy(best_ind))
        
        # Repeat until P' contains N individuals
        while len(new_population) < len(population):

            # Select two parents using a selection algorithm
            first_ind = selection_algorithm(population, maximization)
            second_ind = selection_algorithm(population, maximization)

            if verbose:
                print(f'Selected individuals:\n{first_ind}\n{second_ind}')

            # Apply crossover with probability
            if random.random() < xo_prob:
                offspring1, offspring2 = first_ind.crossover(second_ind)
                if verbose:
                    print(f'Applied crossover')
            else:
                offspring1, offspring2 = deepcopy(first_ind), deepcopy(second_ind)
                if verbose:
                    print(f'Applied replication')
            
            if verbose:
                print(f'Offspring:\n{offspring1}\n{offspring2}')
            
            # Apply mutation to the offspring
            first_new_ind = offspring1.mutation(mut_prob)
            # Insert the mutated individuals into P'
            new_population.append(first_new_ind)

            if verbose:
                print(f'First mutated individual: {first_new_ind}')
            
            if len(new_population) < len(population):
                second_new_ind = offspring2.mutation(mut_prob)
                new_population.append(second_new_ind)
                if verbose:
                    print(f'Second mutated individual: {first_new_ind}')
        
        # Replace P with P'
        population = new_population

        if verbose:
            print(f'Final best individual in generation: {get_best_ind(population, maximization)}')

    # Return the best individual in P
    return get_best_ind(population, maximization)
