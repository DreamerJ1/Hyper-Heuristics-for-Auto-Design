from tabnanny import check
from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptance import MoveAcceptance

class AILTA(MoveAcceptance):
    def __init__(self, hyperHeuristic) -> None:
        super().__init__(hyperHeuristic)

    def createParameters(self):
        self.itteration = 0
        self.itterationLimit = 5
        self.threshhold = [1, 0.1, 50]
        self.threshholdIncreaseRate = [1, 0.1, 10]
        self.threshholdIncreaseItteration = 0
        self.threshholdIncreaseItterationLimit = 5
        self.thresholdUpperBound = [10, 1, 100]
        return self

    def accept(self, oldParetoVector, newParetoVector):
        print(self.itteration, self.threshhold, self.threshholdIncreaseItteration)
        # calculate the pareto dominance
        paretoDominance, paretoDominanceType = self.calculateParetoDominance(oldParetoVector, newParetoVector, self.threshhold)

        print(paretoDominance, paretoDominanceType)
        print("oldParetoVector", oldParetoVector)
        print("newParetoVector", newParetoVector)

        # check if the accuracies provided fall into certain catagories
        if(paretoDominance == True and paretoDominanceType == "better"):
            self.itteration = 0
            self.threshholdIncreaseItteration = 0
            self.threshhold = [1, 0.1, 50]
            return True
        elif(paretoDominance == True and paretoDominanceType == "similar"):
            return True
        else:
            self.itteration += 1
            if(self.itteration >= self.itterationLimit):
                self.threshholdIncreaseItteration += 1
                if(self.threshholdIncreaseItteration >= self.threshholdIncreaseItterationLimit and self.threshhold[0] <= self.thresholdUpperBound[0]):
                    for i in range(len(self.threshhold)):
                        self.threshhold[i] += self.threshholdIncreaseRate[i]
                return False   
            else:
                return False
        return True