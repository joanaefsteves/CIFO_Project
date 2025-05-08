import random

def repair_repr(offspring_repr):
    """
    Repair function to make sure all guest have one and one only seat

    Parameters:
        offspring_repr (dict): Offsprint representation

    Returns:
        dict: The repaired offspring representation

    Raises ValueError if the missing guests does not match number of duplicated guests 
    """

    n_guests = 64

    # The guest list we should have
    all_guests = set(range(n_guests))

    seen = set()
    duplicates = []
    
    # Get duplicates and guests that appear at least once (seen)
    for table, guests in offspring_repr.items():
        for guest in guests:
            if guest in seen:
                duplicates.append((table, guest))
            else:
                seen.add(guest)

    # Get missing guests
    missing_guests = list(all_guests - seen)

    if len(missing_guests) != len(duplicates):
        raise ValueError(f"Number of missing guests does not match number of duplicates.")

    for i, (table, _) in enumerate(duplicates):
        # Remove the duplicate guest from the table
        offspring_repr[table].remove(duplicates[i][1])  
        # Add the missing guest to the table in place of the removed duplicate
        offspring_repr[table].append(missing_guests[i])

    return offspring_repr

# 1. Cycle crossover - Tomás

def cycle_crossover(parent1, parent2):
    """
    Perform cycle crossover between two parents

    Parameters:
        parent1 (dict): The first parent representation
        parent2 (dict): The second parent representation

    Returns:
        tuple: Two repaired offspring representations after performing the crossover
    """

    n_tables = len(parent1)
    offspring1 = {table: [] for table in range(n_tables)}
    offspring2 = {table: [] for table in range(n_tables)}

    # Flatten parents into single lists
    flat_parent1 = [guest for guests in parent1.values() for guest in guests]
    flat_parent2 = [guest for guests in parent2.values() for guest in guests]

    # Initialize offspring as empty lists
    flat_offspring1 = [None] * 64
    flat_offspring2 = [None] * 64

    visited = set()

    # Cycle Crossover logic
    for start_index in range(64):
        if start_index in visited:
            continue

        # Start a new cycle
        index = start_index
        cycle = []

        while index not in visited:
            visited.add(index)
            cycle.append(index)
            index = flat_parent1.index(flat_parent2[index])

        # Assign genes for the cycle
        for idx in cycle:
            flat_offspring1[idx] = flat_parent1[idx]
            flat_offspring2[idx] = flat_parent2[idx]

    # Fill remaining positions with the other parent's genes
    for i in range(64):
        if flat_offspring1[i] is None:
            flat_offspring1[i] = flat_parent2[i]
        if flat_offspring2[i] is None:
            flat_offspring2[i] = flat_parent1[i]

    # Rebuild offspring into table-based representations
    for i in range(64):
        table = i // 8
        offspring1[table].append(flat_offspring1[i])
        offspring2[table].append(flat_offspring2[i])

    return repair_repr(offspring1), repair_repr(offspring2)

# 2. Geometric crossover 

def one_point_overall(c_point, parent1, parent2, n):
    """
    Perform one-point crossover on two parent representations, where the crossover occurs at a single point overall tables

    Parameters:
        c_point (int): The crossover point, a value between 1 and n-1
        parent1 (dict): The first parent representation
        parent2 (dict): The second parent representation
        n (int): The number of tables

    Returns:
        tuple: Two repaired offspring representations after performing the crossover
    """
        
    offspring1 = {}
    offspring2 = {}

    for i in range(n):
        if i < c_point:
            offspring1[i] = parent1[i][:]
            offspring2[i] = parent2[i][:]
        else:
            offspring1[i] = parent2[i][:]
            offspring2[i] = parent1[i][:]

    return repair_repr(offspring1), repair_repr(offspring2)

'''
def one_point_per_table(c_point, parent1, parent2, n):
    """
    Perform one-point crossover on two parent representations, where the crossover occurs at a single point for each table.

    Parameters:
        c_point (int): The crossover point, a value between 1 and n-1
        parent1 (dict): The first parent representation
        parent2 (dict): The second parent representation
        n (int): The number of guests per table

    Returns:
        tuple: Two repaired offspring representations after performing the crossover
    """

    offspring1 = {}
    offspring2 = {}
    
    for i in range(n):
        offspring1[i] = parent1[i][:c_point] + parent2[i][c_point:]
        offspring2[i] = parent2[i][:c_point] + parent1[i][c_point:]

    return repair_repr(offspring1), repair_repr(offspring2)
'''


def crossover(parent1_repr, parent2_repr, crossover_type):
    """
    Performs crossover between two parent representations to generate offspring.
    The crossover type can either be "per_table" or "overall", determining the 
    way the crossover is performed.

    Parameters:
        parent1_repr (dict): The first parent representation
        parent2_repr (dict): The second parent representation
            Both parents must have the same length and type
        crossover_type (str): The type of crossover to be performed

    Returns:
        tuple: Pair of offspring representations (offspring1_repr, offspring2_repr), 
        of the same type as the parents.

     Raises ValueError if the crossover_type is unkown
    """

    n = 8 # Number of tables (also the number of guests per table because they match)

    crossover_point = random.randint(1, n-1)

    crossover_types = {
        "per_table": one_point_per_table,
        "overall": one_point_overall,
        "cycle": cycle_crossover
    }
    
    if crossover_type not in crossover_types:
        raise ValueError(f"Unknown crossover_type: {crossover_type}")
    
    offspring1_repr, offspring2_repr = crossover_types[crossover_type](crossover_point, parent1_repr, parent2_repr, n)

    return offspring1_repr, offspring2_repr
