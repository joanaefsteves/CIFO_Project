
import random
from copy import deepcopy
from SA_problem.seating_arrangement import SASolution

def fitness_proportionate_selection(population: list[SASolution]):
    """
    Selects an individual using fitness-proportionate (roulette wheel) selection.

    Args:
        population (list[SAGASolution]): A list of SeatingArrangement objects.

    Returns:
         A deepcopy of the selected individual.
    """

    fitness_values = [ind.fitness() for ind in population]

    total_fitness = sum(fitness_values)

    # Generate random number between 0 and total
    random_nr = random.uniform(0, total_fitness)
    box_boundary = 0

    # For each individual, check if random number is inside the individual's "box"
    for ind_idx, ind in enumerate(population):
        box_boundary += fitness_values[ind_idx]
        if random_nr <= box_boundary:
            return deepcopy(ind)
        

def ranking_selection(population: list[SASolution]):
    """
    Selects an individual using ranking selection.

    Args:
        population (list[SAGASolution]): A list of SeatingArrangement objects.

    Returns:
         A deepcopy of the selected individual.
    """

    # In maximization, the individual with the highest fitness receives the highest rank.
    sorted_population = sorted(
    population, key=lambda ind: ind.fitness(), reverse=False
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


def tournament_selection(population: list[SASolution],  tournament_size: int = 5):
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

    return deepcopy(max(tournament, key=lambda ind: ind.fitness()))
