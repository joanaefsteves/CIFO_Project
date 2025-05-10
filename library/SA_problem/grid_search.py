
import numpy as np
from itertools import product
from seating_arrangement import SAGASolution
from genetic_algorithms.algorithm import genetic_algorithm

def initialize_population(relations_mtx, pop_size, mutation_function, crossover_function):
    population = []

    for _ in range(pop_size):
        solution = SAGASolution(relations_mtx, mutation_function, crossover_function)
        population.append(solution)

    return population

def grid_search(relations_mtx, 
             pop_size, 
             generations, 
             mutation_functions, 
             crossover_functions, 
             selection_functions, 
             xo_probabilities, 
             mut_probabilities):
    
    results = []

    param_grid = product(
        mutation_functions,
        crossover_functions,
        selection_functions, # if tournment than tournment size is fixed to 5
        xo_probabilities,
        mut_probabilities
        )

    for mutation, crossover, selection, xo_prob, mut_prob in param_grid:
                        
        best_solution = genetic_algorithm(
        initial_population=initialize_population(relations_mtx, pop_size, mutation, crossover),
        max_gen=generations,
        selection_algorithm=selection,
        maximization=True,
        xo_prob=xo_prob,
        mut_prob=mut_prob,
        elitism=True,
        verbose=False
        )

        results.append({
        'mutation': mutation,
        'crossover': crossover,
        'selection': selection,
        'xo probability': xo_prob,
        'mutation probability': mut_prob,
        'fitness': best_solution.fitness(),
        'solution': best_solution
        })
    
    return results

# Fixer hyperparameters:
# tournment size: we can try with different sizes after, if this selection method is selected
# population size: can be around 100 
# generations: can be around 100
