
import numpy as np
from itertools import product
from genetic_algorithms.algorithm import genetic_algorithm


def grid_search(relations_mtx, 
             mutation_functions, 
             crossover_functions, 
             selection_functions, 
             elitism, 
             pop_size = 100, 
             generations = 100
            ):
    
    # Fixed hyperparameters:
    # population size = 100
    # max generations = 100
    # crossover probability = 0.9
    # mutation probability = 0.1
    # when selection method is tournment: tournment size = 5
    
    results = []

    param_grid = product(
        mutation_functions,
        crossover_functions,
        selection_functions, # if select is tournment than tournment size is fixed to 5
        elitism
        )

    for mutation, crossover, selection, elitism in param_grid:
                
        best_solution = genetic_algorithm(
        relations_mtx,
        pop_size=pop_size,
        max_gen=generations,
        selection_algorithm=selection,
        mutation_function=mutation,
        crossover_function=crossover,
        xo_prob=0.9,
        mut_prob=0.1,
        elitism=elitism, 
        verbose=False
        )

        results.append({
        'mutation': mutation,
        'crossover': crossover,
        'selection': selection,
        'elitism': elitism,
        'fitness': best_solution.fitness(),
        'solution': best_solution
        })
    
    return results

# Add code to save fitness history and plot later
# Time each combination takes to run