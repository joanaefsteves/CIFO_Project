import numpy as np
from library.solution import Solution

class SASolution(Solution):

    # Joana:
    # If we have the arrangmement and then create the representation
    # it would be much harder to apply mutation and crossover using this class
    # because they output the representation, which we would have to pass to a arrangement
    # to be able to input it to class SAGASolution
    # which we would then have to convert to a representation again

    def __init__ (self, repr: np.ndarray, relations_mtx: np.ndarray):
        """
        Initializes a seating arrangement.
        
        Parameters:
        - repr: A numpy array of length 64, where each index represents a guest
                and the value is the assigned table (0 to 7).
        - relations_mtx: A 64x64 numpy array with the pairwise relationship scores.
        """

        self.relations_mtx = relations_mtx
        self.nr_tables = 8
        self.nr_guests = 64
        self.fitness = self.fitness_table()

        if repr:
            repr = self.validate_repr(repr)

    def validate_repr(self, repr: np.ndarray) -> np.ndarray:
        """
        Validates the representation:
        - Make sure there are 8 tables and 64 guests
        - Make sure each table has exactly 8 guests (64/8)
        - Make sure all guests are assigned to a table 

        Returns:
        - Validated representation
        
        Raises ValueError if the representation is not valid
        """

        # Check if nº of tables is 8
        if len(repr) != self.nr_tables:
            raise ValueError(f"Representation needs to have {self.nr_tables} tables.")

        if repr.shape[0] != self.nr_guests:
            raise ValueError(f"Representation needs to have {self.nr_guests} guests.")
        
        # Check if all guests are assigned to tables in the range 0 to 7
        if not np.all(np.isin(repr, range(self.nr_tables))):
            raise ValueError("Table assignments must be between 0 and 7.")

        # Check if each table has exactly 8 guests
        for table in range(self.nr_tables):
            if np.sum(repr == table) != self.nr_guests // self.nr_tables:
                raise ValueError(f"Table {table} must have exactly {self.nr_guests // self.nr_tables} guests.")
    
        return repr

    def random_initial_representation(self) -> np.ndarray:

        """
        Generates a random seating arrangement.
        Each guest is randomly assigned to one and only one of the 8 tables.

        Returns:
        - A numpy array with the initial seating arrangement.
        """
        
        # Make array with the index of the tables from 0 to 7
        # Repeat that array by the nº of guests per table (64 guests/8 tables = 8 guests/table)
        repr = np.repeat(np.arange(self.nr_tables), self.nr_guests // self.nr_tables)

        # Shuffle the array so the table assignments are random
        np.random.shuffle(repr)
        
        return repr
    
    def fitness_table(self) -> int:
        """
        Calculates the fitness of the current seating arrangement.
        The fitness is the total sum of relationship scores for all guest pairs seated at the same table.

        Returns:
        - total_fitness (int): The total happiness score of the current arrangement.
        """
        total_fitness = 0

        for table in range(self.nr_tables):

            # Get all guests assigned to the current table
            guests_at_table = np.where(self.repr == table)[0]

            # Sum the relationship scores for all unique pairs at this table
            for i in range(len(guests_at_table)):

                for j in range(i + 1, len(guests_at_table)):
                    total_fitness += self.relations_mtx[guests_at_table[i], guests_at_table[j]]
        
        return total_fitness

    def __str__(self):
        """
        Returns a string showing the guests assigned to each table.
        """
        
        output = ""

        for table in range(self.nr_tables):
            guests_at_table = np.where(self.repr == table)[0]
            output += f"Table {table}: {guests_at_table.tolist()}\n"
            
        return output
    
class SAGASolution(SASolution):

    def __init__ (self, repr: np.ndarray, relations_mtx: np.ndarray, mutation_function, crossover_function):
        
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
            representation=new_repr
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
