from Program.GeneticProgramClasses.FitnessMethod import FitnessMethod

class Raw(FitnessMethod):
    def __init__(self, fitnessMethod) -> None:
        super().__init__(fitnessMethod)

    def calculateFitness(self, pop, output) -> float:
        """
        calculate the raw amount of correct guesses
        """
        ammountCorrect = 0
        for i in range(len(output)):
            if(pop.output[i] == output[i]):
                ammountCorrect += 1
        return ammountCorrect