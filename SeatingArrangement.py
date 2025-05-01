import numpy as np

class SeatingArrangement:

    def __init__ (self, assignments, relations_mtx):
        """
        Initializes a seating arrangement.
        
        Parameters:
        - assignment: A numpy array of length 64, where each index represents a guest
                      and the value is the assigned table (0 to 7).
        - relations_mtx: A 64x64 numpy array with the pairwise relationship scores.
        """

        self.assignments = assignments 
        self.relations_mtx = relations_mtx
        self.representation= self.generate_representation()
        self.fitness = self.fitness_table()
    
    def generate_representation(self):
        """
        Groups guests by tables based on their assignments.
        Returns:
        - A dictionary where keys are table numbers and values are lists of guest IDs.
        """

        tables = {table: [] for table in range(8)}

        for guest_id, table_number in enumerate(self.assignments):
            tables[table_number].append(guest_id)

        return tables
    
    def fitness_table(self):
        """
        Calculates the fitness of the current seating arrangement.
        The fitness is the total sum of relationship scores for all guest pairs seated at the same table.

        Returns:
        - total_fitness (int): The total happiness score of the current arrangement.
        """

        total_fitness=0

        for table in range (8):

            # Get all guests assigned to the current table
            guests_at_table = np.where(self.assignments == table)[0]

            # Sum the relationship scores for all unique pairs at this table
            for i in range(len(guests_at_table)):

                for j in range(i + 1, len(guests_at_table)):
                    a = guests_at_table[i]
                    b = guests_at_table[j]
                    total_fitness += self.relations_mtx[a][b]
        
        return total_fitness

    def __str__(self):
        """
        Returns a string showing the guests assigned to each table.
        """
        
        output = ""

        for table, guests in self.representation.items():
            output += f"Table {table}: {guests}\n"
            
        return output