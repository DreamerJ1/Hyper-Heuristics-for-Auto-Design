import random
from ..SelectionTechniques import SelectionTechniques


class RandomSelection(SelectionTechniques):
    def __init__(self, hyperHeuristic, numToSelect) -> None:
        super().__init__(hyperHeuristic, numToSelect)

    def selection(self):
        """
        Select number of low level heursitics to be applied 
        """
        for i in range(0, self.numToSelect):
            # binary choice between total reconstruction or simply a shift
            choice = random.choice([True, False])
            toBeChanged = {}

            # total reconstruction
            if(choice == True):
                lowLevelHeuristic = random.choice(list(self.hyperHeuristic["completelyChangeGenerationOptions"].keys()))
                toBeChanged.update({lowLevelHeuristic: self.hyperHeuristic["completelyChangeGenerationOptions"][lowLevelHeuristic]})
                print(toBeChanged)
            # shift
            else:
                lowLevelHeuristic = random.choice(list(self.hyperHeuristic["shiftGenerationOptions"].keys()))
                toBeChanged.update({lowLevelHeuristic: self.hyperHeuristic["shiftGenerationOptions"][lowLevelHeuristic]})
                print(toBeChanged)

    