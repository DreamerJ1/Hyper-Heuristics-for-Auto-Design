class FitnessMethod():
    def __init__(self, fitnessMethod) -> None:
        self.fitnessMethod = fitnessMethod

    def calculateFitness(self, pop, output, fitnessCalculationMethod) -> float:
        return 0