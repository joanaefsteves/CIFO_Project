from copy import deepcopy
import random

def swap_mutation(repr):
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
    guest1, guest2 = random.sample(range(len(repr)), 2)

    # Swap the table assignments between selected guests
    new_repr[guest1], new_repr[guest2] = new_repr[guest2], new_repr[guest1]
    
    return new_repr


def inversion_mutation(repr):
    """
    Applies inversion mutation to a solution representation with a given probability.

    Inversion mutation reverses the order of a subsequence of guests in the flattened
    list of all guests and then rebuilds the seating arrangement by assigning them 
    back to tables evenly.

    Parameters:
        repr (dict): The seating arrangement, where each key is a table index (0 to 7),
                     and the value is a list of guest indices assigned to that table.
        mut_prob (float): The probability of performing the inversion mutation.

    Returns:
        dict: A new seating arrangement with a reversed guest subsequence,
              preserving valid assignment (each guest assigned exactly once).
    """

    new_repr = deepcopy(repr)

    # Flatten
    all_guests = [guest for table in sorted(repr.keys()) for guest in repr[table]]

    # Randomly select two positions
    first_idx = random.randint(0, len(all_guests) - 1)
    second_idx = first_idx
    while second_idx == first_idx:
        second_idx = random.randint(0, len(all_guests) - 1)

    if first_idx > second_idx:
        first_idx, second_idx = second_idx, first_idx

     # Invert subsequence
    reversed_subsequence = list(reversed(all_guests[first_idx:second_idx + 1]))
    all_guests = all_guests[:first_idx] + reversed_subsequence + all_guests[second_idx + 1:]

    # Rebuild the dictionary structure with equal guests per table
    guests_per_table = len(all_guests) // len(repr)
    new_repr = {
        i: all_guests[i * guests_per_table:(i + 1) * guests_per_table]
        for i in range(len(repr))
    }

    return new_repr

    
def heuristic_mutation(repr, relationship_mtx):
    """
    Applies a heuristic mutation that swaps a guest with another from a different table
    if the swap increases the overall happiness (affinity).

    Parameters:
        repr (dict): Seating arrangement (table index -> list of guest indices)
        relationship_mtx (np.ndarray): Happiness matrix where [i][j] is the happiness guest i has with guest j
        mut_prob (float): Probability of applying the mutation

    Returns:
        dict: A new seating arrangement with a beneficial swap if found
    """
   
    new_repr = deepcopy(repr)

    all_guests = [(guest, table) for table, guests in repr.items() for guest in guests]
    guest1, table1 = random.choice(all_guests)

    # Evaluate current happiness of guest1 at their table
    current_happiness = sum(
    relationship_mtx[guest1][other] for other in repr[table1] if other != guest1
    )

    best_gain = 0
    best_swap = None

    for table2, guests2 in repr.items():
        if table2 == table1:
            continue

        for guest2 in guests2:
            # Calculate happiness if we swap guest1 and guest2
            temp_table1 = [g if g != guest1 else guest2 for g in repr[table1]]
            temp_table2 = [g if g != guest2 else guest1 for g in repr[table2]]

            new_happiness1 = sum(relationship_mtx[guest1][g] for g in temp_table2 if g != guest1)
            new_happiness2 = sum(relationship_mtx[guest2][g] for g in temp_table1 if g != guest2)
            current_happiness2 = sum(relationship_mtx[guest2][g] for g in repr[table2] if g != guest2)

            total_gain = (new_happiness1 + new_happiness2) - (current_happiness + current_happiness2)

            if total_gain > best_gain:
                best_gain = total_gain
                best_swap = (guest2, table2)

    if best_swap:
        guest2, table2 = best_swap
        new_repr[table1].remove(guest1)
        new_repr[table2].remove(guest2)
        new_repr[table1].append(guest2)
        new_repr[table2].append(guest1)   

    return new_repr


