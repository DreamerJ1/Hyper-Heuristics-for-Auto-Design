from Program.GeneticProgramClasses.FitnessMethod import FitnessMethod

class f1Score(FitnessMethod):
    def __init__(self, fitnessMethod) -> None:
        super().__init__(fitnessMethod)
        self.confusionMatrix = []

    def getIndexFromTerminalSet(self, pop, option) -> int:
        """
        Returns the index of the option in the terminal set
        """
        try:
            return pop.getTerminalSet().index(option.strip(" "))
        except: 
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

        # populate the matrix with the correct values
        for i in range(len(pop.output)):
            self.confusionMatrix[self.getIndexFromTerminalSet(pop, pop.output[i])][self.getIndexFromTerminalSet(pop, output[i])] += 1

    def countForWeightedF1Score(self, pop, output, lenOfCount) -> list:
        """
        Counts the number of times each option is in the input
        """
        count = [0 for i in range(lenOfCount)]
        for i in range(len(output)):
            count[self.getIndexFromTerminalSet(pop, output[i])] += 1
        return count

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

    def calculateFitness(self, pop, output) -> float:
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

        # check which fitness is being asked for and do the calculation
        fitness = 0
        if(self.fitnessMethod == "normal"):
            f1Score = self.createF1ScoreArray(f1Score, recallSum)
            fitness = sum(f1Score) / len(f1Score)
        elif(self.fitnessMethod == "accuracy"):
            tp = 0
            fp = 0
            for i in range(len(self.confusionMatrix)):
                tp += self.confusionMatrix[i][i]
                fp += sum(self.confusionMatrix[i]) - self.confusionMatrix[i][i]
            fitness = tp/(tp+fp)
        elif(self.fitnessMethod == "weightedF1Score"):
            f1Score = self.createF1ScoreArray(f1Score, recallSum)
            count = self.countForWeightedF1Score(pop, output, len(f1Score))
            for i in range(len(f1Score)):
                fitness += (f1Score[i] * count[i])

        # clear the confusion matrix
        self.confusionMatrix = []
        return fitness

    def getFitnessMethod(self) -> str:
        """
        Returns the fitness method
        """
        return self.fitnesMethod

