import random
import statistics
import string
import uuid
from PopTree import PopTree
from treelib import Tree


# FUNCTIONS
def createPopulation(popSize: int, maxDepth: int, functionSet: list, terminalSet: list, arity: list, functionSetChoices: list, method: string):
    """
    Creates a population of trees 
    """
    # create variable to hold populations
    population = []

    # loop that runs for each element in the population
    for i in range(0, popSize):
        # create deep copy of functionset
        functionSetHolder = []
        for j in range(len(functionSet)):
            functionSetHolder.append(functionSet[j])

        # Check which popgrow method is being used
        if(method == "G"):
            tree = PopTree(maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
            tree.growGeneration()
            # check that the tree is valid 
            while(tree.fitness == 0):
                tree = PopTree(maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                tree.growGeneration()
                tree.calculateOutput(trainingInputs)
                calculateFitness(tree, trainingOutputs)
        elif(method == "F"):
            tree = PopTree(maxDepth, functionSetHolder,
                           terminalSet, arity, functionSetChoices)
            tree.fullGeneration()
        elif(method == "H"):
            # check if perfect slit
            if(popSize % 2 == 0):
                # split the pop in half and do each half as either grow or full
                if(i <= int(popSize/2)):
                    tree = PopTree(maxDepth, functionSetHolder,
                                   terminalSet, arity, functionSetChoices)
                    tree.growGeneration()
                else:
                    tree = PopTree(maxDepth, functionSetHolder,
                                   terminalSet, arity, functionSetChoices)
                    tree.fullGeneration()

            else:
                # if its an odd number create half grow and half full and then the left over tree in the middle uses randomly either grow or full
                if(i <= int(popSize/2)):
                    tree = PopTree(maxDepth, functionSetHolder,
                                   terminalSet, arity, functionSetChoices)
                    tree.growGeneration()
                elif(i == int(popSize/2)+1):
                    whichOne = random.randint(0, 1)
                    if(whichOne == 0):
                        tree = PopTree(
                            maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                        tree.growGeneration()
                    else:
                        tree = PopTree(
                            maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                        tree.fullGeneration()
                else:
                    tree = PopTree(maxDepth, functionSetHolder,
                                   terminalSet, arity, functionSetChoices)
                    tree.fullGeneration()

        # Append the tree to the population
        population.append(tree)

    return population

# function to build a confision matrix 
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
    # if tree predicts any clasifer 0 times then its not a valid tree
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
            # weightedPrecision = precision * 0.3
            # weightedRecall = recall * 0.7
            # weightedF1Score[i] = (2 * weightedPrecision * weightedRecall) / (weightedPrecision + weightedRecall)

    # calculate the weighted f1-score
    # f1Score = sum(f1Score) / len(f1Score)
    # return f1Score
    # precision = sum(precisionList) / len(precisionList)
    # return precision
    weightedSums = (count0 * f1Score[0] + count1 * f1Score[1] + count2 * f1Score[2] + count3 * f1Score[3] + count4 * f1Score[4])
    weightedF1Score = (weightedSums / sampleSize) 
    return weightedF1Score
    # weightedSums = (count0 * weightedF1Score[0] + count1 * weightedF1Score[1] + count2 * weightedF1Score[2] + count3 * weightedF1Score[3] + count4 * weightedF1Score[4])
    # weightedF1Score = (weightedSums / sampleSize) 
    # return weightedF1Score
    # weightedPrecision = count0 * precisionList[0] + count1 * precisionList[1] + count2 * precisionList[2] + count3 * precisionList[3] + count4 * precisionList[4]
    # weightedPrecision = weightedPrecision/sampleSize
    # return weightedPrecision

# # raw fitness function
# def calculateFitness(pop, output):
#     ammountCorrect = 0
#     for i in range(len(output)):
#         if(int(pop.output[i]) == int(output[i])):
#             ammountCorrect += 1
#     pop.fitness = ammountCorrect
#     return ammountCorrect / len(output) * 100

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

def checkTerminals(node):
    """
    Checks if the children of a node are only terminals
    """
    for i in range(node.getChildren().__len__()):
        if(node.getChildren()[i].getChildren().__len__() == 0):
            return True
    return False

def getRandomNode(tree: PopTree, maxDepth: int):
    """
    Gets a random node from the tree
    """
    currentDepth = 0
    currentNode = tree.rootNode
    parentNode = currentNode
    parentsParentNode = currentNode
    while(currentDepth <= maxDepth):
        index = functionSet.index(currentNode.getVariable())
        parentsParentNode = parentNode
        parentNode = currentNode
        currentNode = currentNode.getChildAtIndex(random.randint(0, arity[index]-1))
        currentDepth += 1

        # check if we hit a terminal node since we cant go down farther
        if(currentNode.getChildren() == [] or checkTerminals(currentNode)):
            return parentsParentNode, currentDepth-2

    return currentNode, currentDepth-1

# crossover function 
def preformCrossover(popOne, popTwo, depthOne, depthTwo):
    """
    Preforms the crossover of two trees
    """
    # get nodes just above each depth
    nodeOne, depthOneFound = getRandomNode(popOne, depthOne-1)
    nodeTwo, depthTwoFound = getRandomNode(popTwo, depthTwo-1)

    # pick directions that are functional nodes 
    canPickFromForPopOne = []
    canPickFromForPopTwo = []
    for i in range(nodeOne.getChildren().__len__()):
        if(nodeOne.getChildAtIndex(i).getVariable() in functionSet):
            canPickFromForPopOne.append(i)
    for i in range(nodeTwo.getChildren().__len__()):
        if(nodeTwo.getChildAtIndex(i).getVariable() in functionSet):
            canPickFromForPopTwo.append(i)

    # pick one of the valid directions 
    if(canPickFromForPopOne.__len__() > 0):
        popOneDir = canPickFromForPopOne[random.randint(0, canPickFromForPopOne.__len__()-1)]
    else:
        popOneDir = random.randint(0, arity[functionSet.index(nodeOne.getVariable())]-1)
    if(canPickFromForPopTwo.__len__() > 0):
        popTwoDir = canPickFromForPopTwo[random.randint(0, canPickFromForPopTwo.__len__()-1)]
    else:
        popTwoDir = random.randint(0, arity[functionSet.index(nodeTwo.getVariable())]-1)

    # swap the two nodes
    popOneChildHolder = nodeOne.getChildren()
    popTwoChildHolder = nodeTwo.getChildren()
    temp = popOneChildHolder[popOneDir]
    popOneChildHolder[popOneDir] = popTwoChildHolder[popTwoDir]
    popTwoChildHolder[popTwoDir] = temp

    # insert the children back to node
    nodeOne.setChildren(popOneChildHolder)
    nodeTwo.setChildren(popTwoChildHolder)

def preformMutation(pop):
    """
    Preforms mutation on one pop
    """
    # generate random depth and create current node
    depth = random.randint(0, maxDepth-1)
    currentNode = pop.rootNode

    # get random node at depth found 
    currentNode, depthFound = getRandomNode(pop, depth)

    # give the current node a new variable and send it through to the recursive grow algorithm
    whichSet = random.randint(0, 1)
    if(whichSet == 0 and currentNode != pop.rootNode):
        currentNode.setVariable(terminalSet[random.randint(0, len(terminalSet)-1)])
        currentNode.setChildren([])
    else:
        currentNode.setVariable(functionSet[random.randint(0, len(functionSet)-1)])
        currentNode.setChildren([])
        pop.recursivelyCreateGrowTree(currentNode, depthFound)

    return pop

def preformGeneticOperators(population: list, crossover: int, mutationChance: int, tournamentSize: int, output: list):
    """
    preform genetic operators population
    """
    # create new population list
    newPopulation = []

    # loop double the amount of crossover and preform tournament selection
    crossoverPopulations = []
    for i in range(crossover*2):
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

        # del population[population.index(randomSelect[0])]

    # preform the crossover
    for i in range(0, len(crossoverPopulations), 2):
        # get random depth to preform cross over at
        popOneDepth = random.randint(0, maxDepth-1)
        # popTwoDepth = random.randint(0, maxDepth-popOneDepth)
        popTwoDepth = popOneDepth
        preformCrossover(crossoverPopulations[i], crossoverPopulations[i+1], popOneDepth, popTwoDepth)

        # insert the new crossed over populations into the new population
        newPopulation.append(crossoverPopulations[i])
        newPopulation.append(crossoverPopulations[i+1])

    # reproduceduction 
    for i in range(1):
        newPopulation.append(population[i])
        del population[i]

    # preform mutation on new population
    for i in range(int(len(population)*(1-crossoverChance))):
        # check if a mutation is done
        mutationHappen = random.random()
        if(mutationHappen < mutationChance):
            newPopulation.append(preformMutation(population[i]))
        else:
            newPopulation.append(population[i])

    return newPopulation

# key for sorting
def keyForSorting(zipped):
    return zipped[1]

# Sort two lists and return the population
def sortPopulation(population, fitnessList):
    zipped = zip(population, fitnessList)
    ranked = sorted(zipped, key=keyForSorting, reverse=True) 
    population, fitnessList = zip(*ranked)
    return list(population), list(fitnessList)


# recursive function for creating treelib tree form my tree
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


# recrusive calling function
def createTree(pop):
    """ 
    Creates a tree using the given population
    """
    tree = Tree()
    changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
    tree.show()

# recrusive calling function
def saveTreeToFile(pop, fileName):
    """ 
    Creates a tree using the given population
    """
    tree = Tree()
    changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
    tree.save2file("TreeStorage.txt")

# read the data from a file
def readData(file_name):
    """
    Reads data from file and returns a list of lists
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    return data


# function to get max and min value of list index given
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

# check how accurate the output is
def correctnessWithOutput(output, outputResults):
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

    fileName.write("Accuracy: " + str(ammountCorrect / len(output) * 100) + "%\n")


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

# GP sets
terminalSet = ["0", "1", "2", "3", "4"]
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
# populationSeed = 9216
random.seed(populationSeed)
popSize = 100
maxDepth = 5
generationMethod = "Grow"
mutationChance = 0.5
crossoverChance = 0.8
numberOfCrossovers = int((popSize*crossoverChance)/2)
tournamentSize = 4

# Store random numbers
randomFile = open('randomStorage.txt', 'a')
randomFile.write("Number of Divisions:" + str(numOfDivisions) + "\n" +
                 "PopulationSeed: " + str(populationSeed) + "\n" +
                 "Pop Size: " + str(popSize) + "\n" +
                 "Max Tree Depth: " + str(maxDepth) + "\n" +
                 "Population Generation Method: " + generationMethod + "\n"
                 "Mutation chance: " + str(mutationChance) + "\n"
                 "Crossover chance: " + str(crossoverChance) + "\n"
                 "Number of Crossovers to be preformed: " + str(numberOfCrossovers) + "\n"
                 "The Tournament Size: " + str(tournamentSize) + "\n" + "\n")
randomFile.close()


# do 10 runs
allTrainingAccuracies = []
allTestingAccuracies = []
for runs in range(10):
    # create trees
    if(generationMethod == "Grow"):
        print("PRINT GROW TREES")
        population = createPopulation(
            popSize, maxDepth, functionSet, terminalSet, arity, functionSetChoices, "G")

    # # loop population amount of times and print each tree
    # for i in range(len(population)):
    #     print("Tree " + str(i) + ":")
    #     createTree(population[i])

    # loop generation amount of times and apply genetic operators
    fitnessRecord = []
    generations = 100
    for i in range(generations):
        print("Generation: " + str(i+1))

        # calculate the output of each of the trees
        for j in range(len(population)):
            population[j].calculateOutput(trainingInputs)

        # print each of the trees 
        # print(len(population))
        # for i in range(len(population)-1):
        #     if(population[i].fitness != population[i+1].fitness or i == len(population)-2):
        #         createTree(population[i])  
        # createTree(population[0])

        population = preformGeneticOperators(population, numberOfCrossovers, mutationChance, tournamentSize, trainingOutputs)

        # calculate fitnesslist and sort population according 
        fitnessList = []
        for j in range(len(population)):
            fitnessScore = calculateFitness(population[j], trainingOutputs)
            fitnessList.append(fitnessScore)
            population[j].fitness = fitnessScore
        population1, fitnessList1 = sortPopulation(population, fitnessList)
        fitnessRecord.append(fitnessList1[0])
        print("Fitness List: ", fitnessList1[0])

        #  termination cases
        if(fitnessList1[0] > 0.60):
            print("Termination: Fitness Score > 0.6")
            break
        if(len(fitnessRecord) > 50):
            found = 0
            for j in range(0, 6):
                if(fitnessRecord[-1] == fitnessRecord[-j-1]):
                    found += 1
                    print("Fitness Stagnant count: " + str(found))
            if(found >= 5):
                print("Termination: Fitness Stagnant")
                break
    
    # save the generate we ended on 
    gen = i+1

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
    allTrainingAccuracies.append(correctnessWithOutput(population1[0].output, trainingOutputs))

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

    print("\nCorrectness of testingData")
    for j in range(len(population)):
        population[j].calculateOutput(testingInputs)
    allTestingAccuracies.append(correctnessWithOutput(population[0].output, testingOutputs))

    resultsFile.write("\nCorrectness of testing data:\n")
    correctnessWithFileOutput(population[0].output, testingOutputs, resultsFile)
    
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
    resultsFile.write("0: " + str(count0Output) + "\n")
    resultsFile.write("1: " + str(count1Output) + "\n")
    resultsFile.write("2: " + str(count2Output) + "\n")
    resultsFile.write("3: " + str(count3Output) + "\n")
    resultsFile.write("4: " + str(count4Output) + "\n")
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
