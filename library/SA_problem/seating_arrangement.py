import numpy as np
from library.solution import Solution

class SASolution(Solution):

    def __init__ (self, relations_mtx: np.ndarray, repr: dict = None):
        """
        Initializes a seating arrangement.
        
        Parameters:
        - relations_mtx: A 64x64 numpy array with the pairwise relationship scores.
        """

        self.relations_mtx = relations_mtx
        self.nr_tables = 8
        self.nr_guests = 64
        
        if repr:
            repr = self.validate_repr(repr)

        self.repr = repr

    def validate_repr(self, repr: dict) -> dict:
        """
        Validates the representation:
        - Make sure there are 8 tables and 64 guests
        - Make sure each table has exactly 8 guests (64/8)
        - Make sure all guests are assigned to a table and only one table

        Returns:
        - Validated representation
        
        Raises ValueError if the representation is not valid
        """

        # Check if nº of tables is correct 
        if len(repr) != self.nr_tables:
            raise ValueError(f"Representation needs to have {self.nr_tables} tables.")
        
        # Flatten guest list from all tables
        all_guests = [guest for guests in repr.values() for guest in guests]

        # Check if nº of unique guests is correct (and correct range of idx)
        if set(all_guests) != set(range(self.nr_guests)):
            raise ValueError(f"Representation needs to have {self.nr_guests} unique guests.")
        
        # Check that each table has the correct number of guests
        for table_id, guests in repr.items():
            if len(guests) != self.nr_guests // self.nr_tables:
                raise ValueError(f"Table {table_id} must have exactly {self.nr_guests // self.nr_tables} guests.")
    
        return repr

    def random_initial_representation(self) -> dict:

        """
        Generates a random seating arrangement.
        Each guest is randomly assigned to one and only one of the 8 tables.

        Returns:
        - A dictionary where each key is a table index (0 to 7) and the value is a list of guest indices assigned to that table.
        """
        
        # Make array with the index of the tables from 0 to 7
        # Repeat that array by the nº of guests per table (64 guests/8 tables = 8 guests/table)
        repr_array = np.repeat(np.arange(self.nr_tables), self.nr_guests // self.nr_tables)

        # Shuffle the array so the table assignments are random
        np.random.shuffle(repr_array)

        # Tranform the array in a dictionary
        repr = {table_id: [] for table_id in range(self.nr_tables)}
        
        for guest_id, table_id in enumerate(repr_array):
            repr[table_id].append(guest_id)
        
        return repr
    
    def fitness(self) -> int:
        """
        Calculates the fitness of the current seating arrangement.
        The fitness is the total sum of relationship scores for all guest pairs seated at the same table.

        Returns:
        - fitness (int): The total happiness score of the current arrangement.
        """
        fitness = 0

        # Iterate each table's guest list from current dict representation
        for guests in self.repr.values():
            # Interate through each guest at current table
            for i in range(len(guests)):
                # Sum the relationship scores for all unique pairs at the current table
                for j in range(i + 1, len(guests)):
                    fitness += self.relations_mtx[guests[i]][guests[j]]
        
        return fitness

    def __str__(self):
        """
        Returns a string showing the guests assigned to each table.
        """
        
        output = ""

        for table, guests in self.repr.items():
            output += f"Table {table}: {guests}\n"
            
        return output
    
class SAGASolution(SASolution):

    def __init__ (self, relations_mtx: np.ndarray, mutation_function, crossover_function, repr: dict = None):
        
        super().__init__(repr=repr, relations_mtx=relations_mtx)
        
        self.mutation_function = mutation_function
        self.crossover_function = crossover_function

    def mutation(self, mut_prob) -> 'SAGASolution':
        # Apply mutation function to representation
        new_repr = self.mutation_function(self.repr, mut_prob)
        # Create and return individual with mutated representation
        return SAGASolution(
            relations_mtx=self.relations_mtx, 
            mutation_function=self.mutation_function,
            crossover_function=self.crossover_function,
            repr=new_repr
        )

    def crossover(self, other_solution: 'SAGASolution') -> tuple:
        # Apply crossover function to self representation and other solution representation
        offspring1_repr, offspring2_repr = self.crossover_function(self.repr, other_solution.repr)

        # Create and return offspring with new representations
        return (
            SAGASolution(
                relations_mtx=self.relations_mtx, 
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring1_repr
            ),
            SAGASolution(
                relations_mtx=self.relations_mtx, 
                mutation_function=self.mutation_function,
                crossover_function=self.crossover_function,
                repr=offspring2_repr
            )
        )
