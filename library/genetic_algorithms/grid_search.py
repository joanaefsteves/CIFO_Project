
import pandas as pd
from itertools import product
import time
import csv
import os
from .algorithm import genetic_algorithm


def grid_search(relations_mtx, 
             mutation_functions, 
             crossover_functions, 
             selection_functions, 
             elitism, 
             pop_size = 100, 
             generations = 100,
             runs = 30
            ):
    
    """
    Runs a genetic algorithm with different combinations of mutation,
    crossover, selection functions, and elitism over a specified number of trials.
    The performance of each trial and of overall trials for each combination is logged, 
    and the solution from the trial of the combination that achieved the maximum fitness value is returned.
    
    Args:
        relations_mtx (matrix): Relationship score matrix to be used by the genetic algorithm to calculate the fitness.
        mutation_functions (list): A list of mutation functions to try.
        crossover_functions (list): A list of crossover functions to try.
        selection_functions (list): A list of selection functions to try.
        elitism (int): An integer specifying the number of elites, if zero no elitism is applied.
        pop_size (int): The population size for the genetic algorithm (default is 100).
        generations (int): The number of generations to run the genetic algorithm (default is 100).
        runs (int): The number of runs for each combination of parameters (default is 30).
    
    Returns:
        pd.Series: The row from the `log_per_combination.csv` corresponding to the best combination
                   with the highest `max_fitness`.
    """

    
    # Fixed hyperparameters:
    # population size = 100
    # max generations = 100
    # crossover probability = 0.9
    # mutation probability = 0.1
    # when selection method is tournment: tournment size = 3

    log_per_trial = "log_per_trial.csv"

    log_per_combination = "log_per_combination.csv"

    if not os.path.exists(log_per_trial):
        with open(log_per_trial, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "run", "mutation", "crossover", "selection", "elitism",
                "best_solution", "fitness", "trial_run_time"
            ])

    if not os.path.exists(log_per_combination):
        with open(log_per_combination, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "mutation", "crossover", "selection", "elitism",
                "best_final_solution", "avg_fitness", "max_fitness", "min_fitness",
                "total_run_time"
            ])

    param_grid = product(
        mutation_functions,
        crossover_functions,
        selection_functions, # if select is tournment than tournment size is fixed to 5
        elitism
        )

    for mutation, crossover, selection, elitism in param_grid:

        fitness_list = []
        best_solutions = []
        start_time_total = time.time()

        for run in range(1, runs + 1):

            start_time = time.time()
                
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

            trial_run_time = time.time() - start_time
            fitness = best_solution.fitness()

            fitness_list.append(fitness)
            best_solutions.append(best_solution)

            with open(log_per_trial, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    run,
                    mutation.__name__,
                    crossover.__name__,
                    selection.__name__,
                    elitism,
                    best_solution,
                    fitness,
                    round(trial_run_time, 5)
                ])

        run_time_total = time.time() - start_time_total
        avg_fitness = sum(fitness_list) / len(fitness_list)
        max_fitness = max(fitness_list)
        min_fitness = min(fitness_list)
        best_final_solution = best_solutions[fitness_list.index(max(fitness_list))]

        with open(log_per_combination, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    mutation.__name__,
                    crossover.__name__,
                    selection.__name__,
                    elitism,
                    best_final_solution,
                    avg_fitness,
                    max_fitness,
                    min_fitness,
                    round(run_time_total, 5)
                ])
    
    df_per_combination = pd.read_csv(log_per_combination)
    best_combination = df_per_combination.loc[df_per_combination['max_fitness'].idxmax()]

    return best_combination