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
                while(True):
                    lowLevelHeuristic = random.choice(list(self.hyperHeuristic["completelyChangeGenerationOptions"].keys()))
                    if(lowLevelHeuristic != "numberOfOperators" and lowLevelHeuristic != "numberOfTerminationCriterion"):
                        break

                return self.getHeuristic(lowLevelHeuristic, "completelyChangeGenerationOptions"), "total"

            # shift
            else:
                while(True):
                    lowLevelHeuristic = random.choice(list(self.hyperHeuristic["shiftGenerationOptions"].keys()))
                    if(lowLevelHeuristic != "numberOfOperators" and lowLevelHeuristic != "numberOfTerminationCriterion"):
                        break

                return self.getHeuristic(lowLevelHeuristic, "shiftGenerationOptions"), "shift"


    