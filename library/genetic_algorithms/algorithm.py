import numpy as np
from typing import Callable
from copy import deepcopy
from library.SA_problem.seating_arrangement import SASolution
import random


def get_best_individuals(population: list[SASolution], n_ind: int):
    """
    Calculates the fitness of each individual in the population and 
    selects the individuals with the highest fitness.

    Args:
        population (list[Solution]): A list of Solution objects representing the population.

    Returns:
        Solutions: The n individuals with best fitness in the population.
    """

    fitness_list = [ind.fitness() for ind in population]

    # Pair each individual with the corresponding idx
    paired = list(zip(population, fitness_list))

    # Order by fitness
    # Reverse to get descending order
    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)

    # Get n best individuals
    best_individuals = [ind for ind, _ in paired_sorted[:n_ind]]

    return best_individuals

def initialize_population(relations_mtx, pop_size):
    population = []

    for _ in range(pop_size):
        solution = SASolution(relations_mtx)
        population.append(solution)

    return population

def genetic_algorithm(
    relations_mtx: np.ndarray,
    pop_size: int,
    max_gen: int,
    selection_algorithm: Callable,
    mutation_function: Callable,
    crossover_function: Callable,
    xo_prob: float = 0.9,
    mut_prob: float = 0.2,
    elitism: int = 1,
    verbose: bool = False,
):
    """
    Executes a genetic algorithm to optimize a population of solutions.

    Args:
        initial_population (list[Solution]): The starting population of solutions.
        max_gen (int): The maximum number of generations to evolve.
        selection_algorithm (Callable): Function used for selecting individuals.
        mutation_function (Callable): Function used to apply mutation.
        crossover_function (Callable): Function used to apply crossover.
        maximization (bool, optional): If True, maximizes the fitness function; otherwise, minimizes. Defaults to False.
        xo_prob (float, optional): Probability of applying crossover. Defaults to 0.9.
        mut_prob (float, optional): Probability of applying mutation. Defaults to 0.2.
        elitism (bool, optional): If True, carries the best individual to the next generation. Defaults to True.
        verbose (bool, optional): If True, prints detailed logs for debugging. Defaults to False.

    Returns:
        Solution: The best solution found on the last population after evolving for max_gen generations.
    """

    # Initialize a population with N individuals
    population = initialize_population(relations_mtx, pop_size)

    # Repeat until termination condition
    for gen in range(1, max_gen + 1):
        if verbose:
            print(f'-------------- Generation: {gen} --------------')

        # Create an empty population P'
        new_population = []

        # If using elitism, insert best n individuals from P into P'
        if elitism != 0:
            best_individuals = get_best_individuals(population, elitism)
            
            for ind in best_individuals:
                new_population.append(deepcopy(ind))
        
        # Repeat until P' contains N individuals
        while len(new_population) < len(population):

            # Select two parents using a selection algorithm
            first_ind = selection_algorithm(population)
            second_ind = selection_algorithm(population)

            if verbose:
                print(f'Selected individuals:\n{first_ind}\n{second_ind}')

            # Apply crossover with probability
            if random.random() < xo_prob:
                offspring1_repr, offspring2_repr = crossover_function(first_ind.repr, second_ind.repr)
                if verbose:
                    print(f'Applied crossover')
            else:
                offspring1_repr, offspring2_repr = deepcopy(first_ind.repr), deepcopy(second_ind.repr)
                if verbose:
                    print(f'Applied replication')
            
            # Apply mutation with probability
            if random.random() < mut_prob:
                first_new_ind_repr = mutation_function(offspring1_repr)
                second_new_ind_repr = mutation_function(offspring2_repr)
            else:
                first_new_ind_repr = deepcopy(offspring1_repr)
                second_new_ind_repr = deepcopy(offspring2_repr)
            
            first_new_ind = SASolution(relations_mtx, first_new_ind_repr)
            second_new_ind = SASolution(relations_mtx, second_new_ind_repr)

            # Insert first individual into P'
            new_population.append(first_new_ind)

            if verbose:
                print(f'First individual: {first_new_ind}')
            
            # Insert second individual into P' if population is not yet completed
            if len(new_population) < len(population):
                new_population.append(second_new_ind)
                if verbose:
                    print(f'Second individual: {first_new_ind}')
        
        # Replace P with P'
        population = new_population

        final_best = get_best_individuals(population, 1)[0]

        if verbose:
            print(f'Final best individual in generation: {final_best}')

    # Return the best individual in P
    return final_best
