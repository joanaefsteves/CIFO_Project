from copy import deepcopy
import random

# Search + 2 - Matilde

def swap_mutation(repr, mut_prob):
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

    if random.random() <= mut_prob:
        # Flatten
        all_guests = [(guest, table) for table, guests in repr.items() for guest in guests]

        # Randomly select two guests
        guest1, table1 = random.choice(all_guests)
        guest2, table2 = random.choice(all_guests)

        # Guarantee we select two different positions
        while guest1 == guest2:
            guest2, table2 = random.choice(all_guests)

        # Swap tables
        new_repr[table1].remove(guest1)
        new_repr[table2].remove(guest2)
        new_repr[table1].append(guest2)
        new_repr[table2].append(guest1)
    
    return new_repr