from copy import deepcopy
import numpy as np
import random

def swap_mutation(repr: np.ndarray, _)-> np.ndarray:
    """
    Applies swap mutation to a solution repr by swapping the table assignments
    of two randomly selected guests.

    Parameters:
        repr (np.ndarray): A 64-element array representing the seating arrangement, where each 
                            index corresponds to a guest and the value at each index is the table 
                            number (0 to 7) the guest is assigned to.
    
    Returns:
        np.ndarray: New seating arrangement with two guests table assignment swapped.
    """

    new_repr = deepcopy(repr)

    # Randomly select two different guests idxs
    guest1, guest2 = random.sample(range(len(new_repr)), 2)

    # Swap the table assignments between selected guests
    new_repr[guest1], new_repr[guest2] = new_repr[guest2], new_repr[guest1]
    
    return new_repr

def inversion_mutation(repr: np.ndarray, _) -> np.ndarray:
    """
    Applies inversion mutation to a solution representation.

    Inversion mutation reverses the order of a subsequence of guests in the flattened
    list of all guests. 

    Parameters:
        repr (np.ndarray): A 64-element array representing the seating arrangement, where each 
                            index corresponds to a guest and the value at each index is the table 
                            number (0 to 7) the guest is assigned to.
    Returns:
        np.ndarray: A new valid seating arrangement with 1/8 of guests table assigments inversed.
    """

    new_repr = deepcopy(repr)

    # To reduce disruption, limit the inversion to a maximum of 1/8 guests 
    max_size = int(len(new_repr) / 8)

    # Randomly select two positions (without replacement)
    start_idx = random.randint(0, len(new_repr) - max_size)  
    size = random.randint(2, max_size)
    end_idx = start_idx + size - 1

    # Reverse the subsequence
    new_repr[start_idx:end_idx + 1] = new_repr[start_idx:end_idx + 1][::-1]

    return new_repr

def heuristic_mutation(repr:np.ndarray, relationship_mtx:np.ndarray)-> np.ndarray:  
    """
    Applies a heuristic mutation that swaps a guest with another from a different table
    if the swap increases the overall happiness (affinity).

    Parameters:
        repr (np.ndarray): A 64-element array representing the seating arrangement, where each 
                            index corresponds to a guest and the value at each index is the table 
                            number (0 to 7) the guest is assigned to.
        relationship_mtx (np.ndarray): Happiness matrix where [i][j] is the happiness guest i has with guest j

    Returns:
        np.ndarray: A new valid seating arrangement, where each guest is assigned to a table and only one table.
    """
   
    new_repr = deepcopy(repr)

    # Choose a random guest
    guest1= random.randint(0, len(repr) - 1)

    # Get the table the guest is assigned to
    table1= repr[guest1]

    # Get the guests assigned to the same table as guest
    guests_table1 = [i for i in range(len(new_repr)) if new_repr[i] == table1 and i != guest1]

    # Calculate the current happiness of guest1 at their table
    current_happiness1 = sum(relationship_mtx[guest1][other] for other in guests_table1)

    best_gain = 0
    best_guest2 = None

    # Go through all possible swap candidates
    for guest2 in range(len(repr)):
        table2 = new_repr[guest2]

        # Skip same guest or same table (we want a swap between tables)
        is_same_guest = guest2 == guest1
        is_same_table = table2 == table1

        if not is_same_guest and not is_same_table:
            # Simulate the swap
            table1_after = [guest2 if g == guest1 else g for g in range(len(new_repr)) if new_repr[g] == table1]
            table2_after = [guest1 if g == guest2 else g for g in range(len(new_repr)) if new_repr[g] == table2]

            # Calculate new happiness
            new_happiness1 = sum(relationship_mtx[guest1][g] for g in table2_after if g != guest1)
            new_happiness2 = sum(relationship_mtx[guest2][g] for g in table1_after if g != guest2)

            # Calculate old happiness of guest2
            current_happiness2 = sum(relationship_mtx[guest2][g] for g in range(len(new_repr)) if new_repr[g] == table2 and g != guest2)

            total_gain = (new_happiness1 + new_happiness2) - (current_happiness1 + current_happiness2)

            if total_gain > best_gain:
                best_gain = total_gain
                best_guest2 = guest2

    # If we found a good swap, apply it
    if best_guest2 is not None:
        new_repr[guest1], new_repr[best_guest2] = new_repr[best_guest2], new_repr[guest1]

    return new_repr

def misfit_mutation(repr: np.ndarray, relationship_mtx: np.ndarray) -> np.ndarray:

    new_repr = deepcopy(repr)

    nr_tables = 8

    # Group guests by table
    tables = {t: [] for t in range(nr_tables)}
    for guest, table in enumerate(new_repr):
        tables[table].append(guest)

    # For each table, find the guest with the lowest happiness
    least_happy_guests = []

    for table, guests in tables.items():
        min_happiness = 0
        least_happy = None
        for guest in guests:
            happiness = sum(relationship_mtx[guest][other] for other in guests if other != guest)
            if happiness < min_happiness:
                min_happiness = happiness
                least_happy = guest
        if least_happy is not None:
            least_happy_guests.append((least_happy, min_happiness, table))

    # Get two least-happy guests from different tables
    least_happy_guests.sort(key=lambda x: x[1])

    guest1, _, table1 = least_happy_guests[0]
    guest2, _, table2 = least_happy_guests[1]

    # Make sure it's diff tables, if yes swap
    if table1 != table2:
        new_repr[guest1], new_repr[guest2] = new_repr[guest2], new_repr[guest1]

    return new_repr



