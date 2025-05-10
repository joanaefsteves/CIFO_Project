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

def one_point_croosoverl(parent1, parent2):
    """
    Perform one-point crossover on two parent representations, where the crossover occurs at a single point overall tables

    Parameters:
        parent1 (dict): The first parent representation
        parent2 (dict): The second parent representation
    Returns:
        tuple: Two repaired offspring representations after performing the crossover
    """

    n = 8 # Number of tables 

    c_point = random.randint(1, n-1)
        
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