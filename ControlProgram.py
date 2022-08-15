import random
import copy

from Program.GeneticProgram import GeneticProgram

def createRandomGP(generationOptions, data, dataType) -> GeneticProgram:
    """
    Creates random parameters for the genetic program
    """
    # create a deep copy of the dictonary and update each of the options with the random value
    programDict = copy.deepcopy(generationOptions)
    for i in generationOptions:
        if(type(generationOptions[i]) == list):
            if(type(generationOptions[i][0]) == int):
                programDict.update({i: (random.randint(generationOptions[i][0], generationOptions[i][1]))})
            else:
                programDict.update({i: (random.choice(generationOptions[i]))})
        elif(type(generationOptions[i]) == dict):
            if(i == "selectionMethod"):
                operatorDict = {}
                operator = random.choice(list(generationOptions[i]))
                operatorDict.update({operator: round(random.uniform(generationOptions[i][operator][0], generationOptions[i][operator][1]), 0)})
                programDict.update({i: operatorDict})
            elif(i == "fitnessMethod"):
                fitnessMethodDict = {}
                fitnessMethod = random.choice(list(generationOptions[i]))
                fitnessMethodDict.update({fitnessMethod: random.choice(generationOptions[i][fitnessMethod])})
                programDict.update({i: fitnessMethodDict})
            elif(i == "operators"):
                operatorDict = {}
                for j in range(programDict["numberOfOperators"]):
                    operator = random.choice(list(generationOptions[i]))
                    operatorName = operator + str(j)
                    operatorDict.update({operatorName: round(random.uniform(generationOptions[i][operator][0], generationOptions[i][operator][1]), 2)})
                programDict.update({i: operatorDict})
            elif(i == "terminationCondition"):
                operatorDict = {}   
                for j in range(programDict["numberOfTerminationCriterion"]):
                    operator = random.choice(list(generationOptions[i]))
                    operatorDict.update({operator: round(random.uniform(generationOptions[i][operator][0], generationOptions[i][operator][1]), 1)})
                programDict.update({i: operatorDict})

    # create a new genetic program
    return GeneticProgram(programDict, data, dataType)

def readData(dataFile, dataStartLine) -> list:
    """
    Reads the data from the data file
    """
    data = []
    with open(dataFile, "r") as f:
        for i in range(dataStartLine):
            f.readline()
        for line in f:
            data.append(line)
    return data

# create the random seed 
random.seed(random.randint(0, 10000))
# random.seed(9834)

# the design decisions for inisial genetic program
initialGenerationOptions = {
    "populationSize": [10, 50],
    "generations": [10, 50],
    "maxDepth": [2, 5],
    "generationMethod": ["G", "F", "H"],
    "numberOfOperators": [2, 5],
    "numberOfTerminationCriterion": [1, 2],
    "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
    "selectionMethod": {"tournament": [2, 5]},
    "operators": {"crossover": [0.5, 0.9], "mutation": [0.01, 0.2]},
    "terminationCondition": {"maxFitness": [0.5, 0.9]},
}

# read in the data from a specific file and input the line number where @relation is located
data = readData("Datasets/UCI/breast-cancer.arff", 94)
dataType = "c"

# create the genetic program and then run its inisial solution
program = createRandomGP(initialGenerationOptions, data, dataType)
program.runGeneticProgram()