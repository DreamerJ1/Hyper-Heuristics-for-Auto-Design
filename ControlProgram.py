import time
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
seed = random.randint(0, 10000)
# seed = 7958 
random.seed(seed)

# the design decisions for inisial genetic program
initialGenerationOptions = {
    "populationSize": [10, 50],
    "generations": [10, 15],
    "maxDepth": [2, 5],
    "generationMethod": ["G", "F", "H"],
    "numberOfOperators": [2, 5],
    "numberOfTerminationCriterion": [1, 2],
    "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
    "selectionMethod": {"tournament": [2, 5]},
    "operators": {"crossover": [0.5, 0.9], "mutation": [0.01, 0.2]},
    "terminationCondition": {"maxFitness": [0.5, 0.9]},
}

# # the design decisions for inisial genetic program
# initialGenerationOptions = {
#     "populationSize": [100, 500],
#     "generations": [10, 100],
#     "maxDepth": [2, 10],
#     "generationMethod": ["G", "F", "H"],
#     "numberOfOperators": [2, 5],
#     "numberOfTerminationCriterion": [1, 3],
#     "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
#     "selectionMethod": {"tournament": [2, 5]},
#     "operators": {"crossover": [0.5, 0.9], "mutation": [0.01, 0.2]},
#     "terminationCondition": {"maxFitness": [0.5, 0.9]},
# }

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
        "maxDepth": [1, 2],
        # all the options with sub dictionaries will only change the repsect options rather than make a new option
        "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
        "selectionMethod": {"tournament": [1, 2]},
        "operators": {"crossover": [0.1, 0.2], "mutation": [0.05, 0.1]},
        "terminationCondition": {"maxFitness": [0.1, 0.2]},
    }

}

# read in the data from a specific file and input the line number where @relation is located
data = readData("Datasets/UCI/breast-cancer.arff", 94)
# data = readData("Datasets/UCI/primary-tumor.arff", 94)
dataType = "c"

# create the genetic program and then run its inisial solution
program = GeneticProgram(initialGenerationOptions, data, dataType)
program.createRandomGP(initialGenerationOptions, data, dataType)

# print the values the gp got as well as the seed used to get it 
with open("Storage/parameterStorage.txt", 'a') as f: 
    f.write('%s: %s\n' % ("seed", seed))
    for key, value in program.parameters.items(): 
        f.write('%s: %s\n' % (key, value))
    f.write("\n")

program.runGeneticProgramTraining()

"""
The Hyper-Heuristic makes use of selection and move acceptance for its runs, as well as the genetic program for its initial and subsiquint solutions
Options for selection: 
    random
Options for move acceptance:
    acceptAll
    AILTA
Simply swap out the below constructor with the one you want to use
"""
hyperHeuristic = HyperHeuristic(hyperHeuristicOptions, "random", "AILTA")
for i in range(10):
    # create the hyperheuristic and run selection techniques on it
    oldSolution = program.getParameters()
    newSolution = hyperHeuristic.performSelection(oldSolution) 
    print(newSolution)

    # stop program to view what is happening 
    input("\nPress Enter to continue...")

    # save the multi-objective variables from the GP to compare to new solution
    oldBestAccuracy = program.getBestAccuracy()

    # run GP again 
    program.setParameters(newSolution)
    program.runGeneticProgramTraining()

    # testing print
    print("old best accuracy: %s" % oldBestAccuracy)
    print("new best accuracy: %s" % program.getBestAccuracy())

    # perform move acceptance
    accept = hyperHeuristic.performMoveAcceptance(oldBestAccuracy, program.getBestAccuracy())
    if(not accept):
        program.setParameters(oldSolution)
        program.setBestAccuracy(oldBestAccuracy)

    # stop program to view what is happening
    input("\nPress Enter to continue...")

# print the best accuracy and parameters of the final solution
print()
print("Best Parameters:")
print(program.getParameters())
print("Best Accuracy:")
print(program.getBestAccuracy())

# perform run of the testing portion of the data
program.runGeneticProgramTesting()