from tabnanny import check
from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptance import MoveAcceptance

class AILTA(MoveAcceptance):
    def __init__(self, hyperHeuristic) -> None:
        super().__init__(hyperHeuristic)

    def createParameters(self):
        self.itteration = 0
        self.itterationLimit = 5
        self.threshhold = 1
        self.threshholdIncreaseItteration = 0
        self.threshholdIncreaseItterationLimit = 5
        self.thresholdUpperBound = 10
        return self

    def accept(self, oldBestAccuracy, newBestAccuracy):
        print(self.itteration, self.threshhold, self.threshholdIncreaseItteration)
        # check if the accuracies provided fall into certain catagories
        if(newBestAccuracy > oldBestAccuracy):
            self.itteration = 0
            return True
        elif(newBestAccuracy == oldBestAccuracy):
            return True
        else:
            self.itteration += 1
            if(self.itteration >= self.itterationLimit):
                if(abs(newBestAccuracy - oldBestAccuracy) < self.threshhold):
                    self.itteration = 0
                    self.threshholdIncreaseItteration = 0
                    self.threshhold = 1
                    return True
                else:
                    self.threshholdIncreaseItteration += 1
                    if(self.threshholdIncreaseItteration >= self.threshholdIncreaseItterationLimit and self.threshhold <= self.thresholdUpperBound):
                        self.threshhold += 1   
                return False   
            else:
                return False
        return True