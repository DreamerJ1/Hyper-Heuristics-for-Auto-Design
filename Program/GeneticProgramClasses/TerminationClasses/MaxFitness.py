from Program.GeneticProgramClasses.Termination import Termination

class MaxFitness(Termination):
    def __init__(self, maxFitness, typeOfFitness) -> None:
        super().__init__()
        self.maxFitness = maxFitness
        self.typeOfFitness = typeOfFitness

    def terminate(self, bestFitness):
        return bestFitness >= self.maxFitness