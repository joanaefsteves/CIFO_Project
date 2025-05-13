import random
import numpy as np

from collections import Counter

def repair_repr(arr: np.ndarray) -> np.ndarray:
    """
    Ensures that each table has exactly 8 guests by fixing duplicates and missing assignments.

    Parameters:
    arr (np.ndarray): A 64-element array where each index represents a guest and the value 
                        is the table number (0 to 7) to which the guest is assigned.

    Returns:
        np.ndarray: A repaired array with exactly 8 guests per table.
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

def cycle_crossover(parent1:np.ndarray, parent2:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Perform cycle crossover between two parents

    Parameters:
        parent1 (np.ndarray): The first parent representation
        parent2 (np.ndarray): The second parent representation

    Returns:
        tuple[np.ndarray, np.ndarray]: Two repaired offspring representations after performing the crossover
    """

    initial_random_idx= random.randint(0, len(parent1) - 1)

    # Find the indices that belong to the cycle
    cycle_idxs= set()
    current_cycle_idx= initial_random_idx

    while current_cycle_idx not in cycle_idxs:
        cycle_idxs.add(current_cycle_idx)

        # Find the next index in the cycle
        value_parent2 = parent2[current_cycle_idx]
        current_cycle_idx = np.where(parent1 == value_parent2)[0][0]

    # Create the empty offspring representations
    offspring1_repr = np.empty_like(parent1)
    offspring2_repr = np.empty_like(parent2)

    
    for idx in range(len(parent1)):
        if idx in cycle_idxs:
            offspring1_repr[idx] = parent1[idx]
            offspring2_repr[idx] = parent2[idx]
        else:
            offspring1_repr[idx] = parent2[idx]
            offspring2_repr[idx] = parent1[idx]

    return repair_repr(offspring1_repr), repair_repr(offspring2_repr)

# 2. Geometric crossover 

def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one-point crossover on two parent representations, 
    where the crossover occurs at one single point.

    Parameters:
        parent1 (np.ndarray): The first parent representation.
        parent2 (np.ndarray): The second parent representation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two offspring representations after performing the crossover.
    """

    n = len(parent1)  # Number of guests

    c_point = random.randint(1, n-1)  # Random crossover point
    
    # Offspring representations
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)

    # Perform crossover at the selected point
    offspring1[c_point:], offspring2[c_point:] = parent2[c_point:], parent1[c_point:]

    return repair_repr(offspring1), repair_repr(offspring2)


# O outro codigo q tava aqui

# def one_point_crossover(parent1: np.ndarray, parent2: np.ndarray) -> tuple:
#     """
#     Perform one-point crossover on two parent representations, 
#     where the crossover occurs at one single point. Since each index 
#     represents a unique guest, no repair is needed.

#     Parameters:
#         parent1 (np.ndarray): The first parent representation.
#         parent2 (np.ndarray): The second parent representation.

#     Returns:
#         tuple: Two offspring representations after performing the crossover.
#     """

#     n = len(parent1)  # Number of guests

#     c_point = random.randint(1, n - 1)  # Random crossover point

#     # Create new offspring by swapping the table assignments after the crossover point
#     offspring1 = np.concatenate((parent1[:c_point], parent2[c_point:]))
#     offspring2 = np.concatenate((parent2[:c_point], parent1[c_point:]))

#     return repair_tables(offspring1), repair_tables(offspring2)
