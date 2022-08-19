import random

from Program.GeneticProgram import GeneticProgram

from Program.HyperHeuristic import HyperHeuristic

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

# the hyperheuristic options 
hyperHeuristicOptions = {
    "numLowLevelHeuristicsToApply": [1, 3],
    "completelyChangeGenerationOptions": 
    {
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
    },
    "shiftGenerationOptions": 
    {
        "populationSize": [1, 5, 10],
        "generations": [1, 5],
        "maxDepth": [1],
        "generationMethod": [1], # this will change the generation method one to the right or one to the left
        "numberOfOperators": [1], 
        "numberOfTerminationCriterion": [1],
        # all the options with sub dictionaries will only change the repsect options rather than make a new option
        "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
        "selectionMethod": {"tournament": [2, 5]},
        "operators": {"crossover": [0.5, 0.9], "mutation": [0.01, 0.2]},
        "terminationCondition": {"maxFitness": [0.5, 0.9]},
    }

}

# read in the data from a specific file and input the line number where @relation is located
data = readData("Datasets/UCI/breast-cancer.arff", 94)
dataType = "c"

# create the genetic program and then run its inisial solution
program = GeneticProgram(initialGenerationOptions, data, dataType)
program.createRandomGP(initialGenerationOptions, data, dataType)
program.runGeneticProgram()

# create the hyperheuristic and then run it
hyperHeuristic = HyperHeuristic(hyperHeuristicOptions, "random", "acceptAll")
hyperHeuristic.performSelectionAndMoveAcceptance()
