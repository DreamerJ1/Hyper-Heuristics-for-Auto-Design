import copy
import random

from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniques import SelectionTechniques
from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniqueClasses.RandomSelection import RandomSelection
from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniqueClasses.ChoiceFunctionSelection import ChoiceFunctionSelection

from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptance import MoveAcceptance
from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptanceClasses.AILTA import AILTA
from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptanceClasses.AcceptAll import AcceptAll


class HyperHeuristic:
    def __init__(self, hyperHeuristic, selectionTechnique, moveAcceptance) -> None:
        self.hyperHeuristic = hyperHeuristic
        self.numLowLevelHeuristicsToApply = 1

        # creation of the selection technique and move acceptance
        self.selectionTechnique = self.createSelectionTechnique(selectionTechnique)
        self.moveAcceptance = self.createMoveAcceptance(moveAcceptance)

    def createSelectionTechnique(self, selectionTechnique):
        if selectionTechnique == "random":
            return RandomSelection(self.hyperHeuristic, self.numLowLevelHeuristicsToApply)
        elif selectionTechnique == "choiceFunction":
            return ChoiceFunctionSelection(self.hyperHeuristic, self.numLowLevelHeuristicsToApply)
        else:
            raise ValueError("Invalid selection technique")

    def createMoveAcceptance(self, moveAcceptance):
        if moveAcceptance == "acceptAll":
            return AcceptAll(self.hyperHeuristic)
        elif moveAcceptance == "AILTA":
            return AILTA(self.hyperHeuristic).createParameters()
        else:
            raise ValueError("Invalid move acceptance")

    def performSelection(self, currentSolution, currentObjectiveValues, currentTime, currentCPUTime):
        if(type(self.selectionTechnique) == RandomSelection):
            # save heuristic information and print new line for easy understanding
            heuristicToChange, typeOfChange = self.selectionTechnique.selection()
        elif(type(self.selectionTechnique) == ChoiceFunctionSelection):
            heuristicToChange, typeOfChange = self.selectionTechnique.selection(currentObjectiveValues, currentTime, currentCPUTime)
        print("\n")

        newSolution = copy.deepcopy(currentSolution)
        if typeOfChange == "total":
            print("Total reconstruction")
            for i in heuristicToChange:
                newSolution[i] = heuristicToChange[i]

            # reset the number of operators and termination condition
            newSolution["numberOfOperators"] = len(newSolution["operators"])
            newSolution["numberOfTerminationCriterion"] = len(newSolution["terminationCondition"])
        else:
            # check if posative or negative shift
            posativeOrNagative = random.choice([True, False])

            # loop to perform the shifts 
            for i in heuristicToChange:
                if(type(currentSolution[i]) == int):
                    if(posativeOrNagative == True):
                        print("Positive shift")
                        newSolution[i] += heuristicToChange[i]
                    else:
                        if(newSolution[i] <= 1):
                            print("Positive shift")
                            newSolution[i] += heuristicToChange[i]
                        else:
                            print("Negative shift")
                            if(newSolution[i] - heuristicToChange[i] <= 1):
                                newSolution[i] = 2
                            else:
                                newSolution[i] -= heuristicToChange[i]
                elif(i == "generationMethod"):
                    newSolution[i] = heuristicToChange[i]
                elif(i == "selectionMethod"):
                    # get heuristic options based on current selection method and randomly choose one 
                    location = list(currentSolution[i].keys())[0]
                    options = heuristicToChange[i][location]
                    option = random.choice(options) 

                    # either add or subtract from selection method
                    if(posativeOrNagative == True):
                        print("Positive shift")
                        newSolution[i][location] += option
                    else:
                        print("Negative shift")
                        if(newSolution[i][location] - option < 1):
                            newSolution[i][location] = 1
                        else:
                            newSolution[i][location] -= option
                    # newSolution.update({i: round(random.randint(heuristicToChange[list(currentSolution[i].keys())[0]], heuristicToChange[list(currentSolution[i].keys())[0]]), 0)})
                elif(i == "fitnessMethod"):
                    print("Fitness method shift")
                    # get heuristic options based on current selection method and randomly choose one 
                    location = list(currentSolution[i].keys())[0]
                    options = heuristicToChange[i][location]
                    newSolution[i].update({list(currentSolution[i].keys())[0]: random.choice(options)})
                elif(i == "operators"):
                    # loop through all operators and find the same one as the heuristic
                    indexOfSimilarOperators = []
                    for j in range(len(currentSolution[i])):
                        if(list(currentSolution[i].keys())[j][:-1] == list(heuristicToChange[i].keys())[0]):
                            indexOfSimilarOperators.append(j)

                    print(currentSolution[i])
                    print(heuristicToChange[i])
                    print(indexOfSimilarOperators)

                    # randomly pick one of the similar operators and change it
                    indexOfSimilarOperators = random.choice(indexOfSimilarOperators)
                    location = list(currentSolution[i].keys())[indexOfSimilarOperators]
                    option = heuristicToChange[i][location[:-1]]
                    
                    # either add or subtract from selection method
                    if(posativeOrNagative == True):
                        print("Positive shift")
                        newSolution[i][location] += option
                    else:
                        print("Negative shift")
                        newSolution[i][location] -= option
                        
                    newSolution[i][location]= round(newSolution[i][location], 1)
                else:
                    # loop through all criterion and find the same one as the heuristic
                    for j in range(len(currentSolution[i])):
                        if(list(currentSolution[i].keys())[j] == list(heuristicToChange[i].keys())[0]):
                            # get heuristic options based on current selection method and randomly choose one 
                            location = list(currentSolution[i].keys())[j]
                            option = heuristicToChange[i][location]

                            # either add or subtract from selection method
                            if(posativeOrNagative == True):
                                print("Positive shift")
                                newSolution[i][location] += option
                            else:
                                print("Negative shift")
                                newSolution[i][location] -= option
                                
                            newSolution[i][location]= round(newSolution[i][location], 1)

        # print the heuristic to be changed
        print("Selected heuristic to change: ")
        print(heuristicToChange)
        print("\nNew solution: ")
        
        return newSolution

    def performMoveAcceptance(self, oldBestAccuracy, newBestAccuracy):
        """
        Performs move acceptence for the hyper hbheuristic
        """
        accept, term = self.moveAcceptance.accept(oldBestAccuracy, newBestAccuracy)
        if(accept):
            print("Move accepted")
            return True, term
        else: 
            print("Move rejected")
            return False, term

    def updateChoiceFunction(self, currentObjectiveValues, currentTime, currentCPUTime):
        """
        A helper function to envocate the update function of the choice function
        """
        self.selectionTechnique.update(currentObjectiveValues, currentTime, currentCPUTime)

    # Getters and Setters

    def getHyperHeuristic(self):
        return self.hyperHeuristic