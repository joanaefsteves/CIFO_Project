from copy import deepcopy
import random

def swap_mutation(repr):
    """
    Applies swap mutation to a solution representation with a given probability.

    Swap mutation randomly selects two different positions (genes) in the 
    representation and swaps their values.

    Parameters:
        representation (dict): The seating arrangement, where each key is a table index (0 to 7),
                                and the value is a list of guest indices assigned to that table.
        mut_prob (float): The probability of performing the swap mutation.

    Returns:
        dict: A new seating arrangement with two guest assignments swapped between tables.
    """

    new_repr = deepcopy(repr)

    # Flatten
    all_guests = [(guest, table) for table, guests in repr.items() for guest in guests]

    # Randomly select two guests
    guest1, table1 = random.choice(all_guests)
    guest2, table2 = random.choice(all_guests)

    # Guarantee we select two different positions
    while guest1 == guest2:
        guest2, table2 = random.choice(all_guests)

    # Swap tables
    new_repr[table1].append(guest2)
    new_repr[table2].append(guest1)
    new_repr[table1].remove(guest1)
    new_repr[table2].remove(guest2)
    
    return new_repr


def inversion_mutation(repr, mut_prob):
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

    if random.random() <= mut_prob:
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

    
def heuristic_mutation(repr, relationship_mtx, mut_prob):
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

    if random.random() <= mut_prob:
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


