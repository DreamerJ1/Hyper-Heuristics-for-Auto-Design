import random
import numpy as np
import math

from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniques import SelectionTechniques

class ChoiceFunctionSelection(SelectionTechniques):
    def __init__(self, hyperHeuristic, numToSelect) -> None:
        super().__init__(hyperHeuristic, numToSelect)

        # holders for heuristics
        self.currentHeuristic = None
        self.previousHeuristic = None

        # create the f1 dictionaries
        self.objectiveDictionary = {}
        self.timeDictonary = {}
        self.f1Dictionary = {}

        # create the f2 matrices    
        self.objectiveMatrix = [[] for x in range(3)]
        self.timeMatrix = [[] for x in range(3)]
        self.f2Matrix = [[] for x in range(3)]

        # create the independant variables
        self.rankDictionary = {}
        self.cpuTimeDictionary = {}

        # hard set importance parameters
        self.alpha = 0.5
        self.beta = 0.5
        self.delta = 5

        # create the diconaries for saving the data
        self.createDictionaries()

    def createDictionaries(self):
        """
        Creates the dictionaries for the choice function
        """
        # create the dictionaries 
        for highLvlKey in self.hyperHeuristic.keys():
            if(type(self.hyperHeuristic[highLvlKey]) != list):
                self.objectiveDictionary[highLvlKey] = {}
                self.timeDictonary[highLvlKey] = {}
                self.cpuTimeDictionary[highLvlKey] = {}
                self.rankDictionary[highLvlKey] = {}
                self.f1Dictionary[highLvlKey] = {}
                for lowLvlKey in self.hyperHeuristic[highLvlKey].keys():
                    if(lowLvlKey != "numberOfOperators" and lowLvlKey != "numberOfTerminationCriterion"):
                        self.objectiveDictionary[highLvlKey][lowLvlKey] = 0
                        self.timeDictonary[highLvlKey][lowLvlKey] = 0
                        self.cpuTimeDictionary[highLvlKey][lowLvlKey] = 0
                        self.rankDictionary[highLvlKey][lowLvlKey] = 0
                        self.f1Dictionary[highLvlKey][lowLvlKey] = 0

        # create the matrices
        for i in range(3):
            self.objectiveMatrix[i] = [[] for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]
            self.timeMatrix[i] = [[] for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]
            self.f2Matrix[i] = [[] for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]
            for l in range(len(self.objectiveDictionary["completelyChangeGenerationOptions"].keys())):
                self.objectiveMatrix[i][l] = [0 for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]
                self.timeMatrix[i][l] = [0 for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]
                self.f2Matrix[i][l] = [0 for x in range(len(self.hyperHeuristic["completelyChangeGenerationOptions"].keys())-2)]

    def getMaxRank(self):
        """
        Gets the heuristic with the max rank
        """
        max = 0
        countSimilar = 0
        for highLvlKey in self.rankDictionary.keys():
            for lowLvlKey in self.rankDictionary[highLvlKey].keys():
                if(self.rankDictionary[highLvlKey][lowLvlKey] > max):
                    max = self.rankDictionary[highLvlKey][lowLvlKey]
                    heuristic = lowLvlKey
                    type = highLvlKey

                if(max == self.rankDictionary[highLvlKey][lowLvlKey]):
                    countSimilar += 1

        # if all the max values are the same then return a random one
        if(countSimilar >= (len(self.rankDictionary[highLvlKey].keys())*2)):
            if(random.random() > 0.5):
                heuristic = random.choice(list(self.rankDictionary["completelyChangeGenerationOptions"].keys()))
                type = "completelyChangeGenerationOptions"
                self.currentHeuristic = [heuristic, type]
                return heuristic, type
            else:
                heuristic = random.choice(list(self.rankDictionary["shiftGenerationOptions"].keys()))
                type = "shiftGenerationOptions"
                self.currentHeuristic = [heuristic, type]
                return heuristic, type

        # return proper heuristic
        self.currentHeuristic = [heuristic, type]
        return heuristic, type

    def f1(self, herusitic, highLvlKey):
        """
        The first part of the choice function
        """
        I_n = np.sum(self.currentObjective - self.objectiveDictionary[highLvlKey][herusitic])
        T_n = self.currentTime - self.timeDictonary[highLvlKey][herusitic]
        return (I_n / T_n) + (self.alpha * self.f1Dictionary[highLvlKey][herusitic])
    
    def f2(self, heuristic, highLvlKey):
        """
        The second part of the choice function
        """
        if(self.previousHeuristic != None):
            if(self.currentHeuristic[1] == "shiftGenerationOptions" and self.previousHeuristic[1] == "shiftGenerationOptions"):
                I_n = np.sum(self.currentObjective - self.objectiveMatrix[0][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])])
                T_n = self.currentTime - self.timeMatrix[0][list(self.timeDictonary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.timeDictonary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])]
                return (I_n / T_n) + (self.beta * self.f2Matrix[0][list(self.f1Dictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.f1Dictionary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])])
            elif(self.currentHeuristic[1] == "completelyChangeGenerationOptions" and self.previousHeuristic[1] == "completelyChangeGenerationOptions"):
                I_n = np.sum(self.currentObjective - self.objectiveMatrix[1][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])])
                T_n = self.currentTime - self.timeMatrix[1][list(self.timeDictonary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.timeDictonary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])]
                return (I_n / T_n) + (self.beta * self.f2Matrix[1][list(self.f1Dictionary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.f1Dictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])])
            else:
                I_n = np.sum(self.currentObjective - self.objectiveMatrix[2][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])])
                T_n = self.currentTime - self.timeMatrix[2][list(self.timeDictonary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.timeDictonary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])]
                return (I_n / T_n) + (self.beta * self.f2Matrix[2][list(self.f1Dictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.f1Dictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])])
        return 0

    def f3(self, heuristic, highLvlKey):
        """
        The third part of the choice function
        """
        return self.currentCpuTime - self.cpuTimeDictionary[highLvlKey][heuristic]

    def f(self, heuristic, highLvlKey):
        """
        The choice function
        """
        return (self.alpha * self.f1(heuristic, highLvlKey)) + (self.beta * self.f2(heuristic, highLvlKey)) + (self.delta * self.f3(heuristic, highLvlKey))

    def selection(self, currentObjective, currentTime, currentCpuTime):
        """
        Selects a heuristic using the choice function
        """
        # turn objective into matrix and save current values 
        objectiveMatrix = np.matrix(currentObjective)
        self.currentObjective = objectiveMatrix
        self.currentTime = currentTime
        self.currentCpuTime = currentCpuTime

        # loop through each of the low level heuristics and create their ranks 
        for highLvlKey in self.rankDictionary.keys():
            for heuristic in self.rankDictionary[highLvlKey].keys():
                self.rankDictionary[highLvlKey][heuristic] = self.f(heuristic, highLvlKey)

        # output the ranks for checking 
        print(self.rankDictionary)

        # get the max heuristic depending on highest level
        heuristic, type = self.getMaxRank()
        if(type == "shiftGenerationOptions"):
            return self.getHeuristic(heuristic, "shiftGenerationOptions"), "shift"
        else:
            return self.getHeuristic(heuristic, "completelyChangeGenerationOptions"), "total"

    def update(self, currentObjective, currentTime, currentCpuTime):
        """
        Update the dictonaries
        """
        # create matrix for currentObjective
        objectiveMatrix = np.matrix(currentObjective)

        # update the f1 dictionaries
        self.f1Dictionary[self.currentHeuristic[1]][self.currentHeuristic[0]] = self.f1(self.currentHeuristic[0], self.currentHeuristic[1])
        self.objectiveDictionary[self.currentHeuristic[1]][self.currentHeuristic[0]] = objectiveMatrix
        self.timeDictonary[self.currentHeuristic[1]][self.currentHeuristic[0]] = currentTime
        self.cpuTimeDictionary[self.currentHeuristic[1]][self.currentHeuristic[0]] = currentCpuTime

        # update the f2 matrices
        if(self.previousHeuristic != None):
            if(self.currentHeuristic[1] == "shiftGenerationOptions" and self.previousHeuristic[1] == "shiftGenerationOptions"):
                self.f2Matrix[0][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])] = self.f2(self.currentHeuristic[0], self.currentHeuristic[1])
                self.objectiveMatrix[0][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])] = objectiveMatrix    
                self.timeMatrix[0][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.currentHeuristic[0])] = currentTime
            elif(self.currentHeuristic[1] == "completelyChangeGenerationOptions" and self.previousHeuristic[1] == "completelyChangeGenerationOptions"):
                self.f2Matrix[1][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = self.f2(self.currentHeuristic[0], self.currentHeuristic[1])
                self.objectiveMatrix[1][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = objectiveMatrix    
                self.timeMatrix[1][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = currentTime
            else:
                self.f2Matrix[2][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = self.f2(self.currentHeuristic[0], self.currentHeuristic[1])
                self.objectiveMatrix[2][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = objectiveMatrix
                self.timeMatrix[2][list(self.objectiveDictionary["shiftGenerationOptions"].keys()).index(self.previousHeuristic[0])][list(self.objectiveDictionary["completelyChangeGenerationOptions"].keys()).index(self.currentHeuristic[0])] = currentTime

        # set the current to the previous
        self.previousHeuristic = self.currentHeuristic

        