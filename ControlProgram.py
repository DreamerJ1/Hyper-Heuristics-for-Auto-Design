import time
import random
import os

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

def getPretrainedValues(dataFile):
    pass

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
    "terminationCondition": {"maxFitness": {"raw": [0, 100], "f1Score": [0.5, 1]}}
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
        "maxDepth": [1, 2],
        "generationMethod": ["G", "F", "H"],
        # all the options with sub dictionaries will only change the repsect options rather than make a new option
        "fitnessMethod": {"raw": ["holder min"], "f1Score": ["accuracy max", "weightedF1Score max", "normal max"]},
        "selectionMethod": {"tournament": [1, 2]},
        "operators": {"crossover": [0.1, 0.2], "mutation": [0.05, 0.1]},
        "terminationCondition": {"maxFitness": [0.1, 0.2]},
    }
}

# get all files in directory
files = []
for file in os.listdir("C:/Users/Johan/Desktop/UniWork/Postgrad/Github/Hyper-Heuristics-for-Auto-Design/Storage/ClassificationResults/Data"):
    files.append(file)

# # clean file on start of program
# with open("dataFailed.txt", 'w') as f:
#     f.write('')

# file numbers for running 
fileNumber = 7
fileNumberList = [3, 4, 11, 12, 13, 18, 19, 21, 23, 28, 29, 31, 33]

# set if pretrained 
pretrained = True

# loop through all the files
# for fileIndex in range(7, len(files)):
for fileIndex in range(fileNumber-1, fileNumber):
# for fI in fileNumberList:
    print("File: " + files[fileIndex])

    # set which file is to be run 
    dataFileLocation = "Storage/ClassificationResults/Data/" + files[fileIndex]

    # read in the data from a specific file and input the line number where @relation is located
    data = readData(dataFileLocation, 0)
    dataType = "c"

    # get data file name 
    dataFileName = (str(dataFileLocation.split("/")[-1].split(".")[0])) + ".txt"

    # create file to store results and trees
    resultsFile = "Storage/ClassificationResults/Results/" + dataFileName
    pretrainedFile = "Storage/ClassificationResults/Pretrained/" + dataFileName
    treesFile = "Storage/ClassificationResults/TreeStructure/" + dataFileName

    # # save the seed for this run 
    # with open(resultsFile, 'w') as f: 
    #     f.write('%s: %s\n' % ("Starting Seed:", seed))
    #     f.write("\n")

    # # clear trees file
    # with open(treesFile, 'w') as f:
    #     f.write('')

    # run entire program 10 times 
    averageTrainingAccuracy = []
    averageTestingAccuracy = []
    averageTrainingTime = []
    averageTestingTime = []
    averageTrainingComplexity = []
    averageTestingComplexity = []
    allParameters = []
    for i in range(10):
        # check if model should use pretrained or not
        if(pretrained):
            # create GP using pretrained values
            program = GeneticProgram(getPretrainedValues(pretrainedFile), data, dataType)
            exit()
        else:
            # create the genetic program and randomly create its decisions
            program = GeneticProgram(initialGenerationOptions, data, dataType)
            program.createRandomGP(initialGenerationOptions, data, dataType)

        # print the values the gp got as well as the seed used to get it 
        with open("Storage/parameterStorage.txt", 'a') as f: 
            f.write('%s: %s\n' % ("seed", seed))
            for key, value in program.parameters.items(): 
                f.write('%s: %s\n' % (key, value))
            f.write("\n")

        # perform first run of GP to get the initial fitness
        program.runGeneticProgramTraining()

        """
        The Hyper-Heuristic makes use of selection and move acceptance for its runs, as well as the genetic program for its initial and subsiquint solutions
        Options for selection: 
            random
            choiceFunction
        Options for move acceptance:
            acceptAll
            AILTA
        Simply swap out the below constructor with the one you want to use, copy and paste as case sensitive
        """
        hyperHeuristic = HyperHeuristic(hyperHeuristicOptions, "choiceFunction", "AILTA")

        print("Run: ", i+1)

        # run the hyperheuristic
        term = "nonTerm"
        while(term == "nonTerm"):
            # save the multi-objective variables from the GP to compare to old solution
            # the pareto dominance is a minimization equation so turn each max to a min 
            oldParetoVector = [(100 - program.getBestAccuracy()), program.getBestTime(), program.getBestComplexity()]

            # create the hyperheuristic and run selection techniques on it
            oldSolution = program.getParameters()
            currentTime = time.time()
            currentCPUTime = time.process_time()
            newSolution = hyperHeuristic.performSelection(oldSolution, oldParetoVector, currentTime, currentCPUTime) 
            print(newSolution)

            # stop program to view what is happening 
            input("\nPress Enter to continue...")

            # run GP again 
            program.setParameters(newSolution)
            program.runGeneticProgramTraining()

            # save the multi-objective variables from the GP to compare to new solution
            newParetoVector = [(100 - program.getBestAccuracy()), program.getBestTime(), program.getBestComplexity()]

            # perform move acceptance
            accept, term = hyperHeuristic.performMoveAcceptance(oldParetoVector, newParetoVector)
            if(not accept):
                program.setParameters(oldSolution)
                program.setBestAccuracy((100 - oldParetoVector[0]))
                program.setBestTime(oldParetoVector[1])
                program.setBestComplexity(oldParetoVector[2])
            else:
                hyperHeuristic.updateChoiceFunction(oldParetoVector, currentTime, currentCPUTime)

            # stop program to view what is happening
            input("\nPress Enter to continue...")

        # save the results of the run
        averageTrainingAccuracy.append(program.getBestAccuracy())
        averageTrainingTime.append(program.getBestTime())
        averageTrainingComplexity.append(program.getBestComplexity())

        # save info to file
        with open(resultsFile, 'a') as f:
            f.write('%s: %s\n' % ("Run", i+1))
            f.write("\n")
            f.write("Training Results: \n")
            f.write('%s: %s\n' % ("Accuracy", program.getBestAccuracy()))
            f.write('%s: %s\n' % ("Time", program.getBestTime()))
            f.write('%s: %s\n' % ("Complexity", program.getBestComplexity()))
            f.write("\n")

        # perform run of the testing portion of the data
        with open(treesFile, 'a') as f:
            f.write('%s: %s\n' % ("Run: ", i+1))
        program.runGeneticProgramTesting(treesFile)
        with open(treesFile, 'a') as f:
            f.write("\n")

        # save the results of the run  
        averageTestingAccuracy.append(program.getBestAccuracy())
        averageTestingTime.append(program.getBestTime())
        averageTestingComplexity.append(program.getBestComplexity())
        allParameters.append(program.getParameters())

        # save results to file 
        with open(resultsFile, 'a') as f: 
            f.write("Testing Results: \n")
            f.write('%s: %s\n' % ("Accuracy", program.getBestAccuracy()))
            f.write('%s: %s\n' % ("Time", program.getBestTime()))
            f.write('%s: %s\n' % ("Complexity", program.getBestComplexity()))

            f.write("\n")
            f.write("Design Desisions:\n")
            for key, value in program.parameters.items(): 
                f.write('%s: %s\n' % (key, value))
            f.write("\n")
            f.write("_________________________________________________________________________________")
            f.write("\n")

    # save the average results of the run
    with open(resultsFile, 'a') as f:
        f.write("Final Results: \n")
        f.write('%s: %s\n' % ("Training Accuracies", averageTrainingAccuracy))
        f.write('%s: %s\n' % ("Testing Accuracies", averageTestingAccuracy))
        f.write('%s: %s\n' % ("Average Training Accuracy", sum(averageTrainingAccuracy)/len(averageTrainingAccuracy)))
        f.write('%s: %s\n' % ("Average Testing Accuracy", sum(averageTestingAccuracy)/len(averageTestingAccuracy)))
        f.write("\n")

        f.write('%s: %s\n' % ("Training Times", averageTrainingTime))
        f.write('%s: %s\n' % ("Testing Times", averageTestingTime))
        f.write('%s: %s\n' % ("Average Training Time", sum(averageTrainingTime)/len(averageTrainingTime)))
        f.write('%s: %s\n' % ("Average Testing Time", sum(averageTestingTime)/len(averageTestingTime)))
        f.write("\n")

        f.write('%s: %s\n' % ("Training Complexities", averageTrainingComplexity))
        f.write('%s: %s\n' % ("Testing Complexities", averageTestingComplexity))
        f.write('%s: %s\n' % ("Average Training Complexity", sum(averageTrainingComplexity)/len(averageTrainingComplexity)))
        f.write('%s: %s\n' % ("Average Testing Complexity", sum(averageTestingComplexity)/len(averageTestingComplexity)))
        f.write("\n")

        # sort all parameters by averageAccuracy  
        holder = sorted(list(zip(averageTestingAccuracy, allParameters)), key=lambda x: x[0], reverse=True)
        averageTestingAccuracy, allParameters = zip(*holder)
        f.write("Best Design Decisions:\n")
        for key, value in allParameters[0].items():
            f.write('%s: %s\n' % (key, value))
        f.write("\n")

    # save best parameters to file  
    with open(pretrainedFile, 'w') as f:
        holder = sorted(list(zip(averageTestingAccuracy, allParameters)), key=lambda x: x[0], reverse=True)
        averageTestingAccuracy, allParameters = zip(*holder)
        f.write("Best Design Decisions:\n")
        for key, value in allParameters[0].items():
            f.write('%s: %s\n' % (key, value))
        f.write("\n")