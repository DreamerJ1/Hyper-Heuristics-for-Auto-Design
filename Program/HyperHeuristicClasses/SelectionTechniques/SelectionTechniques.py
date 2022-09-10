from dataclasses import replace
import random

class SelectionTechniques:
    def __init__(self, hyperHeuristic, numToSelect) -> None:
        self.hyperHeuristic = hyperHeuristic
        self.numToSelect = numToSelect

    def selection(self):
        pass

    def getHeuristic(self, lowLevelHeuristic, whichLowLevelHeuristicSide):
        """
        Get a heuristic from the dictionary of heuristics
        """
        replacementHeuristic = {}
        for i in self.hyperHeuristic[whichLowLevelHeuristicSide]:
            if(i == lowLevelHeuristic):
                if(type(self.hyperHeuristic[whichLowLevelHeuristicSide][i]) == list):
                    if(type(self.hyperHeuristic[whichLowLevelHeuristicSide][i][0]) == int):
                        replacementHeuristic.update({i: (random.randint(self.hyperHeuristic[whichLowLevelHeuristicSide][i][0], self.hyperHeuristic[whichLowLevelHeuristicSide][i][1]))})
                    else:
                        replacementHeuristic.update({i: (random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))})
                elif(type(self.hyperHeuristic[whichLowLevelHeuristicSide][i]) == dict):
                    if(i == "selectionMethod"):
                        if(whichLowLevelHeuristicSide == "completelyChangeGenerationOptions"):
                            operator = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                            replacementHeuristic.update({operator: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][operator])})
                        else:
                            replacementHeuristic.update({lowLevelHeuristic: self.hyperHeuristic[whichLowLevelHeuristicSide][i]})
                    elif(i == "fitnessMethod"):
                        fitnessMethodDict = {}
                        if(whichLowLevelHeuristicSide == "completelyChangeGenerationOptions"):
                            fitnessMethod = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                            replacementHeuristic.update({fitnessMethod: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][fitnessMethod])})
                        else:
                            replacementHeuristic.update({lowLevelHeuristic: self.hyperHeuristic[whichLowLevelHeuristicSide][i]})
                    elif(i == "operators"):
                        operatorDict = {}
                        if(whichLowLevelHeuristicSide == "completelyChangeGenerationOptions"):
                            replacementHeuristic.update({"numberOfOperators": random.randint(self.hyperHeuristic[whichLowLevelHeuristicSide]["numberOfOperators"][0], self.hyperHeuristic[whichLowLevelHeuristicSide]["numberOfOperators"][1])})
                            for j in range(replacementHeuristic["numberOfOperators"]):
                                operator = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                                operatorName = operator + str(j)
                                operatorDict.update({operatorName: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][operator])})
                        else: 
                            operator = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                            operatorDict.update({operator: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][operator])})
                        replacementHeuristic.update({i: operatorDict})
                    elif(i == "terminationCondition"):
                        operatorDict = {}   
                        if(whichLowLevelHeuristicSide == "completelyChangeGenerationOptions"):
                            replacementHeuristic.update({"numberOfTerminationCriterion": random.randint(self.hyperHeuristic[whichLowLevelHeuristicSide]["numberOfTerminationCriterion"][0], self.hyperHeuristic[whichLowLevelHeuristicSide]["numberOfTerminationCriterion"][1])})
                            for j in range(replacementHeuristic["numberOfTerminationCriterion"]):
                                operator = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                                operatorDict.update({operator: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][operator])})
                        else: 
                            operator = random.choice(list(self.hyperHeuristic[whichLowLevelHeuristicSide][i]))
                            operatorDict.update({operator: random.choice(self.hyperHeuristic[whichLowLevelHeuristicSide][i][operator])})
                        replacementHeuristic.update({i: operatorDict})
        return replacementHeuristic