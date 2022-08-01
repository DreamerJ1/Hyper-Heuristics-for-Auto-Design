import random
from Program.GeneticProgramClasses.SelectionMethod import SelectionMethod

# create class with initializer
class Tournament(SelectionMethod):
    def __init__(self, tournamentSize):
        self.tournamentSize = int(tournamentSize)

    def select(self, population):
        """
        Randomly select the tournament amount out of the population and pass them back
        """
        selected = []
        for i in range(self.tournamentSize):
            selected.append(random.choice(population))
        return selected