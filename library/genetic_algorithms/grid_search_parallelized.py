import pandas as pd
import numpy as np
from itertools import product
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from .algorithm import genetic_algorithm

def single_run(run, mutation, crossover, selection, elitism, relations_mtx, pop_size, generations):
    """
    Executes a single run of the genetic algorithm with the given hyperparameters and returns the best solution and fitnes history.

    Args:
        run (int): The index of the current run.
        mutation (function): The mutation function to be used in the genetic algorithm. 
        crossover (function): The crossover function to be used in the genetic algorithm.
        selection (function): The selection function to be used in the genetic algorithm.
        elitism (int): The number of elits, if zero elitism is not applied.
        relations_mtx (matrix): Relationship score matrix to be used by the genetic algorithm to calculate the fitness.
        pop_size (int): The size of the population for the genetic algorithm.
        generations (int): The number of generations to run the genetic algorithm.

    Returns:
        tuple: A tuple containing the following elements:
            - run (int): The index of the current run.
            - final_best (object): The best solution (individual) found in this run.
            - best_fitness_per_gen (list): A list of best fitness values per generation.
    
    """

    final_best, best_fitness_per_gen = genetic_algorithm(
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

    return run, final_best, best_fitness_per_gen

def grid_search_par(relations_mtx: np.ndarray, 
             mutation_functions: list, 
             crossover_functions: list, 
             selection_functions: list, 
             elitism: list | None = None, 
             pop_size: int = 100, 
             generations: int = 100,
             runs: int = 30,
             verbose: bool = False,
            ) -> list[dict]:
    
    """
    Runs a genetic algorithm with different combinations of mutation, crossover, selection functions, 
    and elitism. Each combination is ran for a predefined number of times.
    
    For each parameter combination and run, the best fitness for each generation at each run
    and the average fitness per generation across all runs are logged.
    
    The function updates information about the combination(s) that achieved the highest average fitness in 
    the final generation and the final solution with best fitness for the given combination(s). 
    We assume there may be cases where more than one combination achieve the same average fitness in 
    the final generation.
    
    Parameters:
        relations_mtx (matrix): Relationship score matrix to be used by the genetic algorithm to calculate the fitness.
        mutation_functions (list): A list of mutation functions to try.
        crossover_functions (list): A list of crossover functions to try.
        selection_functions (list): A list of selection functions to try.
        elitism (list): A list with the number of elites, if zero no elitism is applied.
        pop_size (int): The population size for the genetic algorithm (default is 100).
        generations (int): The number of generations to run the genetic algorithm (default is 100).
        runs (int): The number of runs for each combination of parameters (default is 30).
        verbose (bool): If True, prints detailed logs for debugging. Defaults to False.
    
    Returns:
        list[dict]: A list with one dictionary containing the best-performing combination(s), corresponding average 
        fitness in the final generation and solution with the highest fitness.
    """
    
    # Fixed hyperparameters:
    # population size = 100
    # max generations = 100
    # crossover probability = 0.9
    # mutation probability = 0.1
    # when selection method is tournment: tournment size = 3

    if elitism is None:
        elitism = [1] 

    # Create log files 
    fitness_per_run_log = "fitness_per_run.csv"
    avg_fitness_per_gen_log = "avg_fitness_per_generation.csv"

    if not os.path.exists(fitness_per_run_log):
        with open(fitness_per_run_log, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "run", "mutation", "crossover", "selection", "elitism",
                "generation", "fitness"
            ])

    if not os.path.exists(avg_fitness_per_gen_log):
        with open(avg_fitness_per_gen_log, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "mutation", "crossover", "selection", "elitism",
                "generation", "avg_fitness"
            ])
    
    # Define hyperparameter grid
    param_grid = product(   
        mutation_functions,
        crossover_functions,
        selection_functions,
        elitism
        )

    best_combinations_info = []
    best_avg_fitness = 0

    # Iterate trough each combination of hyperparameters
    for mutation, crossover, selection, elitism in param_grid:

        if verbose: 
            print(f"_________________________________________________________________________________________________________________")
            print(f"Mutation: {mutation.__name__}, Crossover: {crossover.__name__}, Selection: {selection.__name__}, Elits: {elitism}")


        runs_fitness_per_gen = []
        best_solutions = []

         # Parallelize the 30 runs for current combination
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(single_run, run, mutation, crossover, selection, elitism, relations_mtx, pop_size, generations)
                for run in range(1, runs + 1)
            ]

            for future in futures:
                run_idx, final_best, best_fitness_per_gen = future.result()
                runs_fitness_per_gen.append(best_fitness_per_gen)
                best_solutions.append(final_best)

                # Log fitness per generation for current run
                with open(fitness_per_run_log, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    for generation, fitness in enumerate(best_fitness_per_gen, 1):
                        writer.writerow([
                            run_idx,
                            mutation.__name__,
                            crossover.__name__,
                            selection.__name__,
                            elitism,
                            generation,
                            fitness
                        ])

                if verbose:
                    print(f"Run {run_idx}: Best fitness = {final_best.fitness()}")

        # Compute average fitness per generation across all runs
        fitness_df = pd.DataFrame(runs_fitness_per_gen).T
        avg_fitness_per_gen = fitness_df.mean(axis=1)

        # Log average fitness per generation fot current combination
        for generation, avg_fitness in enumerate(avg_fitness_per_gen, 1):
            with open(avg_fitness_per_gen_log, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    mutation.__name__,
                    crossover.__name__,
                    selection.__name__,
                    elitism,
                    generation,
                    avg_fitness
                ])

        # Get average fitness of last generation for current combination
        last_gen_avg_fitness = avg_fitness_per_gen.iloc[-1]

        if verbose:
            print(f"Last generation average fitness = {last_gen_avg_fitness}")

        # Find final solution with highest fitness across all runs
        solution_highest_fitness = max(best_solutions, key=lambda s: s.fitness())
        
        # If current combination has better avg fitness at last gen replace best combination info
        if last_gen_avg_fitness > best_avg_fitness:
            best_avg_fitness = last_gen_avg_fitness
            best_combinations_info = [{ 
                "mutation": mutation.__name__,
                "crossover": crossover.__name__,
                "selection": selection.__name__,
                "elitism": elitism,
                "last_gen_avg_fitness": last_gen_avg_fitness,
                "solution_highest_fitness": solution_highest_fitness
            }]
        # If current combination has equal avg fitness at last gen add best combination info
        # to list of best combinations
        elif last_gen_avg_fitness == best_avg_fitness:
            best_combinations_info.append({  
                "mutation": mutation.__name__,
                "crossover": crossover.__name__,
                "selection": selection.__name__,
                "elitism": elitism,
                "last_gen_avg_fitness": last_gen_avg_fitness,
                "solution_highest_fitness": solution_highest_fitness
            })
 
    return best_combinations_info
