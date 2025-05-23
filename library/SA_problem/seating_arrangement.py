# Group Members - Group P
# Joana Esteves | 20240746 
# Matilde Miguel | 20240549 
# Tomás Figueiredo | 20240941 
# Rita Serra | 20240515

# General
import numpy as np

class SASolution():

    def __init__ (self, relations_mtx: np.ndarray, repr: np.ndarray = None):
        """
        Initializes a seating arrangement.
        
        Parameters:
        - relations_mtx: A 64x64 numpy array with the pairwise relationship scores.
        - repr: A numpy array of length 64, where each index represents a guest

        """

        self.relations_mtx = relations_mtx
        self.nr_tables = 8
        self.nr_guests = 64
        
        if  repr is not None:
            repr = self.validate_repr(repr)
        else:
            repr = self.random_initial_representation()

        self.repr = repr

    def validate_repr(self, repr: np.ndarray) -> np.ndarray:
        """
        Validates the representation:
        - Make sure there are 8 tables and 64 guests
        - Make sure each table has exactly 8 guests (64/8)
        - Make sure all guests are assigned to a table and only one table

        Returns:
        - Validated representation
        
        Raises ValueError if the representation is not valid
        """
       
        # Verify if the representation has the correct number of guests (64)
        if repr.shape[0] != self.nr_guests:
            raise ValueError(f"Representation needs to have {self.nr_guests} guests.")

        # Check if all guests are assigned to tables in the range 0 to 7
        if not np.all(np.isin(repr, range(self.nr_tables))):
            raise ValueError("Table assignments must be between 0 and 7.")

        # Check that each table has the correct number of guests
        for table in range(self.nr_tables):
            if np.sum(repr == table) != (self.nr_guests // self.nr_tables):
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
    
    def fitness(self) -> int:
        """
        Calculates the fitness of the current seating arrangement.
        The fitness is the total sum of relationship scores for all guest pairs seated at the same table.

        Returns:
        - fitness (int): The total happiness score of the current arrangement.
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

