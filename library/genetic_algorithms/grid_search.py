
import numpy as np
import pandas as pd
from itertools import product
from .algorithm import genetic_algorithm


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
        verbose=True
        )

        results.append({
        'mutation': mutation.__name__,
        'crossover': crossover.__name__,
        'selection': selection.__name__,
        'elitism': elitism,
        'fitness': best_solution.fitness(),
        'solution': best_solution.repr
        })

    df_results = pd.DataFrame(results)

    best_combination = df_results.loc[df_results['fitness'].idxmax()]
    
    return df_results, best_combination

# Add code to save fitness history and plot later
# Time each combination takes to run