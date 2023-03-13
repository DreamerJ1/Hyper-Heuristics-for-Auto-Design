import random
import statistics
import uuid
from numpy import number
from treelib import Tree
from onePop import onePop

# FUNCTIONS

def buildByte():
    """
    Builds a byte from a chromosome
    """
    byte = ""
    for i in range(8):
        byte += str(random.randint(0, 1))
    return byte

def createChromosome(chromosomeSize):
    """
    Create a chromosome
    """
    chromosome = []
    for i in range(chromosomeSize):
        chromosome.append(buildByte())
    return chromosome

def createPopulation(popSize, chromosomeSize, expressions) -> list:
    """
    Create the inisial population
    """
    population = []

    # loop through the population size
    for i in range(popSize):
        # create the pop and build its expression and add it to the population
        pop = onePop(createChromosome(chromosomeSize))
        pop.buildExpression(expressions, maxDepth)
        population.append(pop)

    return population

def buildConfusionMatrix(pop, output):
    """
    Builds a confusion matrix
    """
    matrix = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    for i in range(len(pop.output)):
        matrix[int(pop.output[i])][int(output[i])] += 1
    return matrix

def calculateWeightedF1(confusionMatrix):
    """
    Calcuates the weighted f1-score of each of the trees
    """
    # # if tree predicts any clasifer 0 times then its not a valid tree
    for i in range(len(confusionMatrix)):
        if(confusionMatrix[i][i] == 0):
            return 0

    # calculate sum of confusion matrix
    recallSum = [0,0,0,0,0]
    for row in confusionMatrix:
        for i in range(len(row)):
            recallSum[i] += row[i]

    # calculate f1Score for each clasifier
    f1Score = [0, 0, 0, 0, 0]
    precisionList = [0,0,0,0,0]
    recallList = [0,0,0,0,0]
    weightedF1Score = [0, 0, 0, 0, 0]
    for i in range(len(f1Score)):
        if(confusionMatrix[i][i] == 0):
            f1Score[i] = 0
        else:
            precision = confusionMatrix[i][i] / sum(confusionMatrix[i])
            recall = confusionMatrix[i][i] / recallSum[i]
            precisionList[i] = precision
            recallList[i] = recall
            f1Score[i] = (2 * precision * recall) / (precision + recall)

    # accuracy 
    tp = 0
    fp = 0
    for i in range(len(confusionMatrix)):
        tp += confusionMatrix[i][i]
        fp += sum(confusionMatrix[i]) - confusionMatrix[i][i]
    accuracy = tp/(tp+fp)
    return accuracy

    # weightedSums = (count0 * f1Score[0] + count1 * f1Score[1] + count2 * f1Score[2] + count3 * f1Score[3] + count4 * f1Score[4])
    # weightedF1Score = (weightedSums / sampleSize) 
    # return weightedF1Score

# the fitness function
def calculateFitness(pop, output):
    """
    Calculates the fitness of each of the trees
    """
    # build matrix 
    confusionMatrix = buildConfusionMatrix(pop, output)

    # calculate weighted f1-score for each class
    f1Score = calculateWeightedF1(confusionMatrix)
    pop.fitness = f1Score
    return f1Score

def performGeneticOperators(population, mutationChance, numberOfCrossovers, tournamentSize, output):
    """
    preform genetic operators population
    """
    # create a new population
    newPopulation = []

    # loop double the amount of crossover and preform tournament selection
    crossoverPopulations = []
    for i in range(numberOfCrossovers*2):
        randomSelect = []
        for j in range(tournamentSize):
            randomSelect.append(population[random.randint(0, len(population)-1)])

        # get fitness for each of the indexes
        fitnessList = []
        for pop in randomSelect:
            fitness = calculateFitness(pop, output)
            pop.fitness = fitness
            fitnessList.append(fitness)

        # sort the randomSelected population and select the best
        randomSelect, fitnessList = sortPopulation(randomSelect, fitnessList)
        crossoverPopulations.append(randomSelect[0].deepCopy())

    # preform crossover 
    for i in range(0, len(crossoverPopulations), 2):
        # randomly pick two swapping points (+ 1 to take the right side of the index)
        popOneChromeIndex = random.randint(0, len(crossoverPopulations[i].chromosome)-1) 
        popTwoChromeIndex = random.randint(0, len(crossoverPopulations[i+1].chromosome)-1)

        # # swap the chromosomes
        crossoverPopulations[i].chromosome[popOneChromeIndex:], crossoverPopulations[i+1].chromosome[popTwoChromeIndex:] = crossoverPopulations[i+1].chromosome[popTwoChromeIndex:], crossoverPopulations[i].chromosome[popOneChromeIndex:]

        # add swapped chrosovers to population
        newPopulation.append(crossoverPopulations[i])
        newPopulation.append(crossoverPopulations[i+1])

    # preform mutation
    for i in range(int(len(population)*(1-crossoverChance))+1):
        for j in range(len(population[i].chromosome)):
            if(random.random() < mutationChance):
                population[i].chromosome[j] = buildByte()
        newPopulation.append(population[i])

    return newPopulation

def changeTreeAndPrint(currentNode, id, isRoot, tree):
    """
    Changes the tree to a treelib tree and print it
    """
    # create id to store parent
    parentId = id
    id = str(uuid.uuid4())

    # break recursion
    if(len(currentNode.getChildren()) == 0):
        tree.create_node(str(currentNode.variable), id, parent=parentId)
        return

    # create the root node
    if(isRoot):
        tree.create_node(currentNode.variable, id)
        for i in range(len(currentNode.getChildren())):
            changeTreeAndPrint(currentNode.getChildAtIndex(i), id, False, tree)
    else:
        # create a node for current node
        tree.create_node(currentNode.variable, id, parent=parentId)
        for i in range(len(currentNode.getChildren())):
            changeTreeAndPrint(currentNode.getChildAtIndex(i), id, False, tree)

def createTree(pop):
    """ 
    Creates a tree using the given population
    """
    # check that there is not only a root node
    if(len(pop.rootNode.getChildren()) == 0):
        print(pop.rootNode.getVariable())
        return

    # otherwise create tree
    tree = Tree()
    changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
    tree.show()

def keyForSorting(zipped):
    """
    Key for sorting
    """
    return zipped[1]

def sortPopulation(population, fitnessList):
    """
    Sort two lists and return the population
    """
    zipped = zip(population, fitnessList)
    ranked = sorted(zipped, key=keyForSorting, reverse=True) 
    population, fitnessList = zip(*ranked)
    return list(population), list(fitnessList)

def correctnessWithOutput(output, outputResults):
    """
    Check how accurate the output is
    """
    outputList = []
    ammountCorrect = 0
    for i in range(len(output)):
        outputPrint = "[" + str(output[i]) + ", " + str(outputResults[i]) + "]"
        if(int(output[i]) == int(outputResults[i])):
            ammountCorrect += 1
            outputPrint += "x"
        outputList.append(outputPrint)

    print(ammountCorrect, len(output))
    print(outputList)
    print(ammountCorrect/len(output)*100)

    ans = ammountCorrect / len(output) * 100
    return ans

def getMaxMin(data, index):
    """
    Gets the max and min value of the given index
    """
    max = float(data[0][index])
    min = float(data[0][index])
    for i in range(1, len(data)):
        if(float(data[i][index]) > max):
            max = float(data[i][index])
        if(float(data[i][index]) < min):
            min = float(data[i][index])
    return int(max), int(min)

def readData(file_name):
    """
    Reads data from file and returns a list of lists
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data

def correctnessWithFileOutput(output, outputResults, fileName):
    outputList = []
    ammountCorrect = 0
    for i in range(len(output)):
        outputPrint = "[" + str(output[i]) + ", " + str(outputResults[i]) + "]"
        if(int(output[i]) == int(outputResults[i])):
            ammountCorrect += 1
            outputPrint += "x"
        outputList.append(outputPrint)
    
    # print the output 10 at a time
    for i in range(0, len(outputList), 10):
        fileName.write(str(outputList[i:i+10]) + "\n")

    fileName.write("Accuracy: " + str(round(ammountCorrect / len(output) * 100, 2)) + "%\n")

# recrusive calling function
def saveTreeToFile(pop, fileName):
    """ 
    Creates a tree using the given population
    """
    tree = Tree()
    changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
    tree.save2file("TreeStorage.txt")

# MAIN

# read processed.cleveland.data file
data = readData('Data/processed.cleveland.data')

# split data into input and output lists
inputs = []
outputs = []
for row in data:
    inputs.append(row[:-1])
    outputs.append(row[-1])

# split input and output into respective training and testing sets
trainingInputs = inputs[:int(len(inputs) * 0.8)]
trainingOutputs = outputs[:int(len(outputs) * 0.8)]
testingInputs = inputs[int(len(inputs) * 0.8):]
testingOutputs = outputs[int(len(outputs) * 0.8):]

# remove rows with unknown values
storeIndex = []
for i in range(len(trainingInputs)):
    if("?" in trainingInputs[i]):
        storeIndex.append(i)
for i in storeIndex:
    del trainingInputs[i]
    del trainingOutputs[i]

count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
sampleSize = len(trainingInputs)
for i in trainingOutputs:
    if(i == "0"):
        count0 +=1 
    elif(i == "1"):
        count1 +=1
    elif(i == "2"):
        count2 +=1
    elif(i == "3"):
        count3 +=1
    elif(i == "4"):
        count4 +=1

print("0: ", count0)
print("1: ", count1)
print("2: ", count2)
print("3: ", count3)
print("4: ", count4)
print("Sample size: ", sampleSize)

# remove a 0's for smoothing of data
count = 0
for i in range(len(trainingOutputs)):
    if(int(trainingOutputs[i]) == 0): 
        del trainingOutputs[i]
        del trainingInputs[i]
        count += 1
    
    if(count == 70 or i == len(trainingOutputs)-1):
        break

count0 = 0
count1 = 0
count2 = 0
count3 = 0
count4 = 0
sampleSize = len(trainingOutputs)
for i in trainingOutputs:
    if(i == "0"):
        count0 +=1 
    elif(i == "1"):
        count1 +=1
    elif(i == "2"):
        count2 +=1
    elif(i == "3"):
        count3 +=1
    elif(i == "4"):
        count4 +=1

print("0: ", count0)
print("1: ", count1)
print("2: ", count2)
print("3: ", count3)
print("4: ", count4)
print("Sample size: ", sampleSize)

# 0 = age - random
# 1 = sex - 0, 1
# 2 = cp - 1, 2, 3, 4
# 3 = trestbps - random
# 4 = chol - random
# 5 = fbs - 0, 1
# 6 = restecg - 0, 1, 2
# 7 = thalach - random
# 8 = exang - 0, 1
# 9 = oldpeak - random
# 10 = slope - 1, 2, 3
# 11 = ca - 0, 1, 2, 3
# 12 = thal - 3, 6, 7

# Dictionary 
expressions = [["o4", "e", "e", "e", "e"], ["o3", "e", "e", "e"], ["o2", "e", "e"], ["v"]]
dictionary = {"o4" : ["cp", "ca"], "o3" : ["thal", "slope", "restecg", "age", "trestbps", "chol", "thalach", "oldpeak"], "o2" : ["sex", "fbs", "exang"], "v" : ["0", "1", "2", "3", "4"]}

functionSet = ["age", "sex", "cp", "trestbps", "chol", "fbs",
               "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
functionSetChoices = [[], [0, 1], [1, 2, 3, 4], [], [], [0, 1],
                      [0, 1, 2], [], [0, 1], [], [1, 2, 3], [0, 1, 2, 3], [3, 6, 7]]
arity = [0, 2, 4, 0, 0, 2, 3, 0, 2, 0, 3, 4, 3]

# create random seed to subdivide each of the unknown divisions
numOfDivisions = 2

# loop through arity
for i in range(len(arity)):
    # check if arity has already
    if(arity[i] == 0):
        # false so get max and min at this index
        max1, min = getMaxMin(trainingInputs, i)

        # loop number of division times and create random number between max and min and store each in holder array
        holder = []
        count = 1
        for j in range(numOfDivisions):
            value = int(((j+1)*(1/(numOfDivisions+1)))*max1)

            # if value is already in holder break
            if(value in holder):
                count = j+1
                break

            holder.append(value)
            min = holder[j]
            count += 1

        # store holder and count
        arity[i] = count
        functionSetChoices[i] = holder

print("Arity: ", arity)
print("Function Set", functionSet)
print("Function Set Choices: ", functionSetChoices)

# GP perameters
populationSeed = random.randint(0, 10000)
populationSeed = 7892
random.seed(populationSeed)
popSize = 100
chromosomeSize = 10
maxDepth = 5
mutationChance = 0.1
crossoverChance = 0.8
numberOfCrossovers = int((popSize*crossoverChance)/2)
tournamentSize = 4

# Store random numbers
randomFile = open('randomStorage.txt', 'a')
randomFile.write("Number of Divisions: " + str(numOfDivisions) + "\n" +
                 "PopulationSeed: " + str(populationSeed) + "\n" +
                 "Pop Size: " + str(popSize) + "\n" +
                 "Chromosome Size: " + str(chromosomeSize) + "\n" +
                 "Max Depth: " + str(maxDepth) + "\n" +
                 "Mutation chance: " + str(mutationChance) + "\n"
                 "Crossover chance: " + str(crossoverChance) + "\n"
                 "Number of Crossovers to be preformed: " + str(numberOfCrossovers) + "\n"
                 "The Tournament Size: " + str(tournamentSize) + "\n" + "\n")
randomFile.close()

# do 10 runs
allTrainingAccuracies = []
allTestingAccuracies = []
for runs in range(10):
    # create population
    population = createPopulation(popSize, chromosomeSize, expressions)

    # loop generations amount of times and preform genetic operators 
    generations = 100
    fitnessRecord = []
    for i in range(generations):
        print("Generation: ", str(i+1))

        # calculate outputs for each population 
        for pop in population:
            pop.calculateOutput(trainingInputs, trainingOutputs, dictionary, functionSet, functionSetChoices)

        # calculate fitnesslist and sort population according 
        fitnessList = []
        for j in range(len(population)):
            fitnessScore = calculateFitness(population[j], trainingOutputs)
            fitnessList.append(fitnessScore)
            population[j].fitness = fitnessScore
        population1, fitnessList1 = sortPopulation(population, fitnessList)
        fitnessRecord.append(fitnessList1[0])
        print("Fitness List: ", fitnessList1[0])

        # preform the getnetic operators 
        population = performGeneticOperators(population, mutationChance, numberOfCrossovers, tournamentSize, trainingOutputs)

        # update the tree structures with the new chromosome 
        for pop in population:
            pop.buildExpression(expressions, maxDepth)

    

    # save the generate we ended on 
    gen = i+1

    # calculate final output after new expressions have been created
    for pop in population:
        pop.calculateOutput(trainingInputs, trainingOutputs, dictionary, functionSet, functionSetChoices)

    # calculate fitnesslist and sort population according 
    fitnessList = []
    for i in range(len(population)):
        fitnessScore = calculateFitness(population[i], trainingOutputs)
        fitnessList.append(fitnessScore)
        population[i].fitness = fitnessScore
    population1, fitnessList1 = sortPopulation(population, fitnessList)
    print("Fitness List: ", fitnessList1)

    # couunt how many 0's in trainingOutput
    print("Correctness Of trainingData")
    allTrainingAccuracies.append(round(correctnessWithOutput(population1[0].output, trainingOutputs), 2))
    print(population1[0].getChromosome())

# for i in range(len(population)):
#     createTree(population[i])

# open results file and write results
    resultsFile = open('results.txt', 'a')
    resultsFile.write("\n" + "Run " + str(runs+1) + " solution found on the " + str(gen) + " generation" + "\n")
    resultsFile.write("Fitness of Best Population:\n")
    resultsFile.write(str(fitnessList1[0]) + "\n")
    resultsFile.write("\n")
    resultsFile.write("Correctness of training data:\n")
    correctnessWithFileOutput(population1[0].output, trainingOutputs, resultsFile)

    print("\nNumber of each guess:")
    count0Output = 0
    count1Output = 0
    count2Output = 0
    count3Output = 0
    count4Output = 0
    sampleSize = len(trainingOutputs)
    for i in population[0].output:
        if(i == 0):
            count0Output +=1 
        elif(i == 1):
            count1Output +=1
        elif(i == 2):
            count2Output +=1
        elif(i == 3):
            count3Output +=1
        elif(i == 4):
            count4Output +=1

    # write each count and sample size to file 
    resultsFile.write("0: " + str(count0Output) + " " + str(count0) + "\n")
    resultsFile.write("1: " + str(count1Output) + " " + str(count1) + "\n")
    resultsFile.write("2: " + str(count2Output) + " " + str(count2) + "\n")
    resultsFile.write("3: " + str(count3Output) + " " + str(count3) + "\n")
    resultsFile.write("4: " + str(count4Output) + " " + str(count4) + "\n")
    resultsFile.write(str(population1[0].getChromosome()[:chromosomeSize]) + "\n")

    print("\nCorrectness of testingData")
    for pop in population:
        pop.buildExpression(expressions, maxDepth)

    for pop in population:
        pop.calculateOutput(testingInputs, testingOutputs, dictionary, functionSet, functionSetChoices)

    # calculate fitnesslist and sort population according 
    fitnessList = []
    for i in range(len(population)):
        fitnessScore = calculateFitness(population[i], testingOutputs)
        fitnessList.append(fitnessScore)
        population[i].fitness = fitnessScore
    population1, fitnessList1 = sortPopulation(population, fitnessList)
    allTestingAccuracies.append(round(correctnessWithOutput(population1[0].output, testingOutputs), 2))

    resultsFile.write("\nCorrectness of testing data:\n")
    correctnessWithFileOutput(population[0].output, testingOutputs, resultsFile)

    count0T = 0
    count1T = 0
    count2T = 0
    count3T = 0
    count4T = 0
    sampleSize = len(testingOutputs)
    for i in testingOutputs:
        if(i == "0"):
            count0T +=1 
        elif(i == "1"):
            count1T +=1
        elif(i == "2"):
            count2T +=1
        elif(i == "3"):
            count3T +=1
        elif(i == "4"):
            count4T +=1
    
    count0Output = 0
    count1Output = 0
    count2Output = 0
    count3Output = 0
    count4Output = 0
    sampleSize = len(testingOutputs)
    for i in population[0].output:
        if(i == 0):
            count0Output +=1 
        elif(i == 1):
            count1Output +=1
        elif(i == 2):
            count2Output +=1
        elif(i == 3):
            count3Output +=1
        elif(i == 4):
            count4Output +=1

    # write each count and sample size to file 
    resultsFile.write("0: " + str(count0Output) + " " + str(count0T) + "\n")
    resultsFile.write("1: " + str(count1Output) + " " + str(count1T) + "\n")
    resultsFile.write("2: " + str(count2Output) + " " + str(count2T) + "\n")
    resultsFile.write("3: " + str(count3Output) + " " + str(count3T) + "\n")
    resultsFile.write("4: " + str(count4Output) + " " + str(count4T) + "\n")
    resultsFile.write(str(population1[0].getChromosome()[:chromosomeSize]) + "\n")
    resultsFile.close()

    # open tree storage and save the run
    treeStorage = open('treeStorage.txt', 'a')
    treeStorage.write("\n" + "Run " + str(runs+1) + " solution found on the " + str(gen) + " generation" + "\n")
    treeStorage.close()
    saveTreeToFile(population1[0], resultsFile)

# calculate the best, average and standard deviation of all accuracies and add them to results file 
resultsFile = open('results.txt', 'a')
resultsFile.write("\n" + "All Training Accuracies: " + "\n")
resultsFile.write(str(allTrainingAccuracies) + "\n")
resultsFile.write("Average Accuracy of all training runs: " + str(sum(allTrainingAccuracies)/len(allTrainingAccuracies)) + "\n")
resultsFile.write("Standard Deviation of all training runs: " + str(statistics.stdev(allTrainingAccuracies)) + "\n")
resultsFile.write("Best Accuracy of all training runs: " + str(max(allTrainingAccuracies)) + "\n")
resultsFile.write("\n" + "All Testing Accuracies: " + "\n")
resultsFile.write(str(allTestingAccuracies) + "\n")
resultsFile.write("Average Accuracy of all testing runs: " + str(sum(allTestingAccuracies)/len(allTestingAccuracies)) + "\n")
resultsFile.write("Standard Deviation of all testing runs: " + str(statistics.stdev(allTestingAccuracies)) + "\n")
resultsFile.write("Best Accuracy of all testing runs: " + str(max(allTestingAccuracies)) + "\n")
resultsFile.close()