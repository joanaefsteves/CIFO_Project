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
        "overall": one_point_overall
    }
    
    if crossover_type not in crossover_types:
        raise ValueError(f"Unknown crossover_type: {crossover_type}")
    
    offspring1_repr, offspring2_repr = crossover_types[crossover_type](crossover_point, parent1_repr, parent2_repr, n)

    return offspring1_repr, offspring2_repr
