from Program.GeneticProgramClasses.FitnessMethod import FitnessMethod

class f1Score(FitnessMethod):
    def __init__(self) -> None:
        self.confusionMatrix = []

    def getIndexFromTerminalSet(self, pop, option) -> int:
        """
        Returns the index of the option in the terminal set
        """
        return pop.getTerminalSet().index(option)

    def buildConfusionMatrix(self, pop, output) -> list:
        """
        Build the confusion matrix to preform calculations on it
        """
        # loop the length of the terminal set and create the matrix
        for i in range(len(pop.getTerminalSet())):
            self.confusionMatrix.append([])
            # loop the length of the terminal set and create the matrix
            for j in range(len(pop.getTerminalSet())):
                self.confusionMatrix[i].append(0)

        print(self.getIndexFromTerminalSet(pop, pop.output[0]))

        # populate the matrix with the correct values
        for i in range(len(pop.output)):
            self.confusionMatrix[self.getIndexFromTerminalSet(pop, pop.output[i])][self.getIndexFromTerminalSet(pop, output[i])] += 1

    def countForWeightedF1Score(self, output):
        """
        Counts the number of times each option is in the input
        """
        

    def createF1ScoreArray(self, f1Score, recallSum) -> list:
        """
        create the array used for each of the calculations 
        """
        for i in range(len(f1Score)):
            if(self.confusionMatrix[i][i] == 0):
                f1Score[i] = 0
            else:
                precision = self.confusionMatrix[i][i] / sum(self.confusionMatrix[i])
                recall = self.confusionMatrix[i][i] / recallSum[i]
                f1Score[i] = (2 * precision * recall) / (precision + recall)
        return f1Score

    def calculateFitness(self, pop, output, fitnessCalculationMethod) -> float:
        """
        Overrides the calculateFitness method in the FitnessMethod class
        """
        # build confusion matrix for pop
        self.buildConfusionMatrix(pop, output)

        # # chech that pop has predicted atleast each option once
        # for i in range(len(self.confusionMatrix)):
        #     if(self.confusionMatrix[i][i] == 0):
        #         return 0

        # create needed arrays
        f1Score = []
        recallSum = []
        for i in range(len(self.confusionMatrix[0])):
            f1Score.append(0)
            recallSum.append(0)

        # calculate sum of confusion matrix
        for row in self.confusionMatrix:
            for i in range(len(row)):
                recallSum[i] += row[i]

        print(f1Score, recallSum)

        print(fitnessCalculationMethod["f1Score"])

        # check which fitness is being asked for and do the calculation
        fitness = 0
        if(fitnessCalculationMethod["f1Score"] == "normal"):
            pass
        elif(fitnessCalculationMethod["f1Score"] == "accuracy"):
            tp = 0
            fp = 0
            for i in range(len(self.confusionMatrix)):
                tp += self.confusionMatrix[i][i]
                fp += sum(self.confusionMatrix[i]) - self.confusionMatrix[i][i]
            fitness = tp/(tp+fp)
        elif(fitnessCalculationMethod["f1Score"] == "weightedF1Score"):
            self.createF1ScoreArray(f1Score, recallSum)
            for i in range(len(f1Score)):
                fitness += f1Score[i]

        # clear the confusion matrix
        self.confusionMatrix = []
        return fitness