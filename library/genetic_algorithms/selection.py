
import random
from copy import deepcopy
from SA_problem.seating_arrangement import SAGASolution

def fitness_proportionate_selection(population: list[SAGASolution], maximization: bool):
    """
    Selects an individual using fitness-proportionate (roulette wheel) selection.

    Args:
        population (list[SAGASolution]): A list of SeatingArrangement objects.
        maximization (bool): If True, maximizes the fitness function; otherwise, minimizes.

    Returns:
         A deepcopy of the selected individual.
    """

    if maximization:
        fitness_values = [ind.fitness() for ind in population]
    else:
        # Minimization: Use the inverse of the fitness value
        # Lower fitness should have higher probability of being selected
        fitness_values = [1 / ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)

    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    box_boundary = 0

    # For each individual, check if random number is inside the individual's "box"
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)
        

def ranking_selection(population: list[SAGASolution], maximization: bool):
    """
    Selects an individual using ranking selection.

    Args:
        population (list[SAGASolution]): A list of SeatingArrangement objects.
        maximization (bool): If True, maximizes the fitness function; otherwise, minimizes.

    Returns:
         A deepcopy of the selected individual.
    """

    # In maximization, the individual with the highest fitness receives the highest rank.
    # In minimization, the individual with the lowest fitness receives the highest rank.
    if maximization:
        sorted_population = sorted(
        population, key=lambda ind: ind.fitness(), reverse=False
        )
    else:
         sorted_population = sorted(
        population, key=lambda ind: ind.fitness(), reverse=True
        )
    
    # Create an ascending ranking from 1 to N=len(population)+1
    rank_scores= range(1, len(sorted_population) + 1) 

    total_rank_scores = sum(rank_scores)

    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_rank_scores)
    box_boundary = 0

    for ind_idx, ind in enumerate(sorted_population):
        box_boundary += rank_scores[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)


def tournament_selection(population: list[SAGASolution], tournament_size: int, maximization: bool):
    """
    Selects an individual using tournament selection.

    Args:
        population (list[SAGASolution]): A list of SeatingArrangement objects.
        tournament_size (int): Number of individuals to participate in the tournament.
        maximization (bool): If True, maximizes the fitness function; otherwise, minimizes.

    Returns:
         A deepcopy of the selected individual.
    """
    
    # random.sample-> selects, randomly, the individuals without replacement 
    tournament= random.sample(population, k=tournament_size)

    # For maximization, return the individual with the highest fitness
    if maximization:
        return deepcopy(max(tournament, key=lambda ind: ind.fitness))
    
    # For minimization, return the individual with the lowest fitness
    else:
        return deepcopy(min(tournament, key=lambda ind: ind.fitness))