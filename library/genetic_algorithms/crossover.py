import random
import numpy as np

from collections import Counter

def repair_tables(arr: np.ndarray) -> np.ndarray:
    """
    Ensures that each table has exactly 8 guests by fixing duplicates and missing assignments.
    """
    table_counts = Counter(arr)
    missing = []
    excess = []

    # Identify which tables are under- or overrepresented
    for table in range(8):
        count = table_counts.get(table, 0)
        if count < 8:
            missing.extend([table] * (8 - count))
        elif count > 8:
            excess.extend([table] * (count - 8))

    if len(missing) != len(excess):
        raise ValueError("Mismatch in number of missing and excess table assignments.")

    arr = arr.copy()
    # Replace excess table assignments with missing ones
    for i in range(len(arr)):
        if excess:
            if table_counts[arr[i]] > 8:
                replacement = missing.pop()
                table_counts[arr[i]] -= 1
                arr[i] = replacement
                table_counts[replacement] += 1

    return arr

def repair_repr(offspring_repr):

    """
    Repair function to ensure all guests have one and only one seat in the seating arrangement.

    Parameters:
        offspring_repr (np.ndarray): A 64-element array representing the seating arrangement,
                                      where each index corresponds to a guest, and the value at
                                      each index is the table number (0 to 7) assigned to the guest.

    Returns:
        np.ndarray: The repaired seating arrangement with no missing or duplicate guests.

    Raises ValueError if the number of missing guests does not match the number of duplicated guests.
    """


    n_guests = 64

    # The guest list we should have
    all_guests = set(range(n_guests))

    # Find duplicates and guests that appear at least once
    seen = set()
    duplicates = []
    
    for guest in range(n_guests):
        if guest in seen:
            duplicates.append(guest)
        else:
            seen.add(guest)

    # Get missing guests (don't appear at all)
    missing_guests = list(all_guests - seen)

    if len(missing_guests) != len(duplicates):
        raise ValueError(f"Number of missing guests does not match number of duplicates.")

    # Replace duplicates with missing guests
    for i, guest in enumerate(duplicates):
        offspring_repr[guest] = missing_guests[i]

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

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """
    Perform one-point crossover on two parent representations, 
    where the crossover occurs at one single point.

    Parameters:
        parent1 (np.ndarray): The first parent representation.
        parent2 (np.ndarray): The second parent representation.

    Returns:
        tuple: Two offspring representations after performing the crossover.
    """

    n = len(parent1)  # Nr of guests

    c_point = random.randint(1, n-1)  # Random crossover point
    
    # Offspring representations
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)

    # Perform crossover at the selected point
    offspring1[c_point:], offspring2[c_point:] = parent2[c_point:], parent1[c_point:]

    return repair_repr(offspring1), repair_repr(offspring2)

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """
    Perform one-point crossover on two parent representations, 
    where the crossover occurs at one single point. Since each index 
    represents a unique guest, no repair is needed.

    Parameters:
        parent1 (np.ndarray): The first parent representation.
        parent2 (np.ndarray): The second parent representation.

    Returns:
        tuple: Two offspring representations after performing the crossover.
    """

    n = len(parent1)  # Number of guests

    c_point = random.randint(1, n - 1)  # Random crossover point

    # Create new offspring by swapping the table assignments after the crossover point
    offspring1 = np.concatenate((parent1[:c_point], parent2[c_point:]))
    offspring2 = np.concatenate((parent2[:c_point], parent1[c_point:]))

    return repair_tables(offspring1), repair_tables(offspring2)
