from copy import deepcopy
import random

# TO BE ADAPTED
# WE HAVE A NUMPY ARRAY NOT A STRING/LIST

def swap_mutation(representation, mut_prob):
    """
    Applies swap mutation to a solution representation with a given probability.

    Swap mutation randomly selects two different positions (genes) in the 
    representation and swaps their values. This operator is commonly used for 
    permutation-based representations but works for any list or string.

    The function preserves the type of the input representation: if the input is 
    a string, the output will also be a string; if it's a list, the output will 
    remain a list.

    Parameters:
        representation (str or list): The solution to mutate.
        mut_prob (float): The probability of performing the swap mutation.

    Returns:
        str or list: A new solution with two genes swapped, of the same type as the input.
    """

    new_representation = deepcopy(representation)

    if random.random() <= mut_prob:
        # Strings are not mutable. Let's convert temporarily to a list
        if isinstance(representation, str):
            new_representation = list(new_representation)

        first_idx = random.randint(0, len(representation) - 1)

        # To guarantee we select two different positions
        second_idx = first_idx
        while second_idx == first_idx:
            second_idx = random.randint(0, len(representation) - 1)

        new_representation[first_idx] = representation[second_idx]
        new_representation[second_idx] = representation[first_idx]

        # If representation was a string, convert list back to string
        if isinstance(representation, str):
            new_representation = "".join(new_representation)
    
    return new_representation