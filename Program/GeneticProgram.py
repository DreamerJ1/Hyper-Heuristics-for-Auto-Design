import copy
import random
import uuid

from treelib import Tree as TreeLib

from Program.GeneticProgramClasses.Tree import Tree 
from Program.GeneticProgramClasses.TreeClasses.DecisionTree import DecisionTree

from Program.GeneticProgramClasses.FitnessMethod import FitnessMethod
from Program.GeneticProgramClasses.FitnessMethodClasses.Raw import Raw
from Program.GeneticProgramClasses.FitnessMethodClasses.f1Score import f1Score

from Program.GeneticProgramClasses.SelectionMethod import SelectionMethod
from Program.GeneticProgramClasses.SelectionMethodClasses.Tournament import Tournament

from Program.GeneticProgramClasses.Operators import Operators
from Program.GeneticProgramClasses.OperatorClasses.Crossover import Crossover
from Program.GeneticProgramClasses.OperatorClasses.Mutation import Mutation

class GeneticProgram:
    def __init__(self, inisialGenerationOptions: dict, data, dataType) -> None:
        self.parameters = self.createRandomGP(inisialGenerationOptions, data, dataType)
        self.terminalSet = []
        self.functionSet = []
        self.functionSetChoices = []
        self.arity = []
        self.inputHandling(data, dataType)
        self.fitnessMethodSelection()
        self.SelectionMethodSelection()
        self.operatorSelection()

    def createRandomGP(self, generationOptions, data, dataType):
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
        return programDict
        

    def inputHandling(self, data, dataType) -> None:
        """
        Handles the data used for genetic program
        """
        if(dataType == "r"):
            pass
        elif(dataType == "c"):
            dataFound = False
            dataList = []
            for i in range(len(data)):
                if("@attribute" in data[i]):
                    # get the functional sets variables
                    startOfAttrabuteName = data[i].find(" ")
                    endOfAttrabuteName = data[i].find("{")
                    attributeName = data[i][startOfAttrabuteName + 1:endOfAttrabuteName].strip(" ").strip("'").strip("\"")

                    # safe the choices for the attribute
                    startOfChoices = data[i].find("{")
                    endOfChoices = data[i].find("}")
                    choices = data[i][startOfChoices + 1:endOfChoices].split(",")

                    # check if the attribute is in the terminal or functionSetChoices
                    if(attributeName == 'Class' or "@data" in data[i+1]):
                        self.terminalSet = choices
                    else:
                        self.functionSetChoices.append(choices)
                        self.functionSet.append(attributeName)
                        self.arity.append(len(choices))
                        
                elif("@data" in data[i]):
                    dataFound = True

                # if data has been found next values in the data file are the data
                elif(dataFound):
                    dataList.append(data[i].split(","))

            print("Terminal set:", self.terminalSet)
            print("Function set:", self.functionSet)
            print("Function set choices:", self.functionSetChoices)
            print("Arity:", self.arity)

        # clean up the data
        for line in dataList: 
            for i in range(len(line)):
                line[i] = line[i].strip("\n")

        # split data into input and output lists
        self.inputs = []
        self.outputs = []
        for row in dataList:
            self.inputs.append(row[:-1])
            self.outputs.append(row[-1])
        
        # split input and output into respective training and testing sets
        self.trainingInputs = self.inputs[:int(len(self.inputs) * 0.8)]
        self.trainingOutputs = self.outputs[:int(len(self.outputs) * 0.8)]
        self.testingInputs = self.inputs[int(len(self.inputs) * 0.8):]
        self.testingOutputs = self.outputs[int(len(self.outputs) * 0.8):]

        # remove rows with unknown values
        storeIndex = []
        for i in range(len(self.trainingInputs)):
            if("?" in self.trainingInputs[i]):
                storeIndex.append(i)
        for i in storeIndex:
            del self.trainingInputs[i]
            del self.trainingOutputs[i]

    def createPopulation(self, popSize, maxDepth, functionSet, terminalSet, arity, functionSetChoices, method) -> list:
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
                tree = DecisionTree(maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                tree.growGeneration()

                # # check that the tree is valid 
                # while(tree.fitness == 0):
                #     tree = DecisionTree(maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                #     tree.growGeneration()
                #     tree.calculateOutput(trainingInputs)
                #     calculateFitness(tree, trainingOutputs)
            elif(method == "F"):
                tree = DecisionTree(maxDepth, functionSetHolder,
                            terminalSet, arity, functionSetChoices)
                tree.fullGeneration()
            elif(method == "H"):
                # check if perfect slit
                if(popSize % 2 == 0):
                    # split the pop in half and do each half as either grow or full
                    if(i <= int(popSize/2)):
                        tree = DecisionTree(maxDepth, functionSetHolder,
                                    terminalSet, arity, functionSetChoices)
                        tree.growGeneration()
                    else:
                        tree = DecisionTree(maxDepth, functionSetHolder,
                                    terminalSet, arity, functionSetChoices)
                        tree.fullGeneration()

                else:
                    # if its an odd number create half grow and half full and then the left over tree in the middle uses randomly either grow or full
                    if(i <= int(popSize/2)):
                        tree = DecisionTree(maxDepth, functionSetHolder,
                                    terminalSet, arity, functionSetChoices)
                        tree.growGeneration()
                    elif(i == int(popSize/2)+1):
                        whichOne = random.randint(0, 1)
                        if(whichOne == 0):
                            tree = DecisionTree(
                                maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                            tree.growGeneration()
                        else:
                            tree = DecisionTree(
                                maxDepth, functionSetHolder, terminalSet, arity, functionSetChoices)
                            tree.fullGeneration()
                    else:
                        tree = DecisionTree(maxDepth, functionSetHolder,
                                    terminalSet, arity, functionSetChoices)
                        tree.fullGeneration()

            # Append the tree to the population
            population.append(tree)

        return population

    def fitnessMethodSelection(self):
        """
        Selects the fitness method to be used
        """
        # select which fitness method 
        if(list(self.parameters["fitnessMethod"].keys())[0] == "raw"):
            # set the fitness direction to either min or max
            fitnessMethod = self.parameters["fitnessMethod"]["raw"]
            fitnessMethod = fitnessMethod.split(" ")
            self.parameters["fitnessMethodDirection"] = fitnessMethod[1]

            self.fitnessMethod = Raw(fitnessMethod[0])
        elif(list(self.parameters["fitnessMethod"].keys())[0] == "f1Score"):  
            # set the fitness direction to either min or max
            fitnessMethod = self.parameters["fitnessMethod"]["f1Score"]
            fitnessMethod = fitnessMethod.split(" ")
            self.parameters["fitnessMethodDirection"] = fitnessMethod[1]

            self.fitnessMethod = f1Score(fitnessMethod[0])  

    def SelectionMethodSelection(self):
        """
        Selects the selection method to be used
        """
        if(list(self.parameters["selectionMethod"].keys())[0] == "tournament"):
            self.selectionMethod = Tournament(self.parameters["selectionMethod"]["tournament"])

    def operatorSelection(self):
        """
        Selects the operator to be used
        """
        # create array to store all the operator objects to be used 
        self.operators = []
        for operatorDic in self.parameters["operators"]:
            operator = operatorDic[:-1]
            chance = self.parameters["operators"][operatorDic]
            if(operator == "crossover"):
                self.operators.append(Crossover(chance))
            elif(operator == "mutation"):
                self.operators.append(Mutation(chance))

    def preformGeneticOperators(self, population, output) -> list:
        """
        Preform the genetic operators for the genetic program
        Note: The operators all perform reproduction on the remaining population they did not interact with as to
        maintain the population size
        """
        # create new population 
        newPopulation = []
        # loop through the possible operators and preform each as needed 
        for i in range(len(self.operators)):
            # perform crossover 
            if(type(self.operators[i]) == Mutation):
                # loop entire population, check if mutation happens and perform it
                for j in range(len(population)):
                    if(random.random() < self.operators[i].getMutationChance()):
                        newPopulation.append(self.operators[i].performOperation(population[j]))
                    else:
                        newPopulation.append(population[j])

            elif(type(self.operators[i]) == Crossover):
                # perform selection 
                crossoverPopulation = []
                numCrossover = int(self.operators[i].getCrossoverChance() * len(population))
                for j in range((numCrossover)):
                    # select individuals for possible crossover
                    selected = self.selectionMethod.select(population)

                    # get the fitness of each selected and select best fitness
                    selectedFitness = []
                    for pop in selected:
                        fitness = self.fitnessMethod.calculateFitness(pop, output)
                        pop.fitness = fitness
                        selectedFitness.append(fitness)

                    # get the best fitness and add it to the list to be crossovered
                    selected, selectedFitness = self.sortPopulation(selected, selectedFitness, self.parameters["fitnessMethodDirection"])
                    crossoverPopulation.append(selected[0].deepCopy())

                # perform crossover on the selected individuals
                for j in range(0, len(crossoverPopulation), 2):
                    child1, child2 = self.operators[i].performOperation(crossoverPopulation[i], crossoverPopulation[i+1])
                    newPopulation.append(child1)
                    newPopulation.append(child2)

                # perform reproduction on the rest of the population
                for j in range(0, len(population) - len(crossoverPopulation)):
                    newPopulation.append(population[j].deepCopy())

            # add the new population to the old population
            population = newPopulation
            newPopulation = []
        
        return population

    def changeTreeAndPrint(self, currentNode, id, isRoot, tree) -> None:
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
                self.changeTreeAndPrint(currentNode.getChildAtIndex(i), id, False, tree)
        else:
            # create a node for current node
            tree.create_node(currentNode.variable, id, parent=parentId)
            for i in range(len(currentNode.getChildren())):
                self.changeTreeAndPrint(currentNode.getChildAtIndex(i), id, False, tree)

    def createTree(self, pop) -> None:
        """ 
        Creates a tree using the given population
        """
        tree = TreeLib()
        self.changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
        tree.show()

    def keyForSorting(self, zipped):
        """
        key for sorting
        """
        return zipped[1]

    def sortPopulation(self, population, fitnessList, fitnessDirection):
        """
        Sort two lists and return the population
        """
        if(fitnessDirection == "max"):
            population, fitnessList = zip(*sorted(zip(population, fitnessList), key=self.keyForSorting, reverse=True))
        else:
            population, fitnessList = zip(*sorted(zip(population, fitnessList), key=self.keyForSorting))

        return list(population), list(fitnessList)

    def correctnessWithOutput(self, output, outputResults):
        """
        Check how accurate the output is
        """
        outputList = []
        ammountCorrect = 0
        for i in range(len(output)):
            outputPrint = "[" + str(output[i]) + ", " + str(outputResults[i]) + "]"
            if(output[i] == outputResults[i]):
                ammountCorrect += 1
                outputPrint += "x"
            outputList.append(outputPrint)

        print(ammountCorrect, len(output))
        print(outputList)
        print(ammountCorrect/len(output)*100)

        ans = ammountCorrect / len(output) * 100
        return ans

    def runGeneticProgram(self) -> None:
        """
        The main function of the genetic program
        """
        # store the parameters of the gp in a text file 
        with open("Storage/parameterStorage.txt", 'a') as f: 
            for key, value in self.parameters.items(): 
                f.write('%s: %s\n' % (key, value))
            f.write("\n")

        # create the initial population
        population = self.createPopulation(self.parameters["populationSize"], self.parameters["maxDepth"], self.functionSet, 
        self.terminalSet, self.arity, self.functionSetChoices, self.parameters["generationMethod"])
        
        # loop generation amount of times
        for i in range(self.parameters["generations"]):
            print("Generation: " + str(i+1))

            # calculate output
            for j in range(len(population)):
                population[j].calculateOutput(self.trainingInputs)
                
            # preform genetic operations
            population = self.preformGeneticOperators(population, self.trainingOutputs)

            population, fitnessList = self.sortPopulation(population, [self.fitnessMethod.calculateFitness(pop, self.trainingOutputs) for pop in population], self.parameters["fitnessMethodDirection"])
            print(fitnessList   )

        # sort the population by their fitness
        population1, fitnessList1 = self.sortPopulation(population, [self.fitnessMethod.calculateFitness(pop, self.trainingOutputs) for pop in population], self.parameters["fitnessMethodDirection"])
        print("Best fitness: " + str(fitnessList1[0]))
        print("Best program: ")
        self.createTree(population1[0])

        print("Total accuracy:")
        self.correctnessWithOutput(population1[0].output, self.trainingOutputs)

    # GETTERS AND SETTERS
    def getParameters(self):
        return self.parameters
    
    def setParameters(self, parameters):
        self.parameters = parameters

    def getFunctionSet(self):
        return self.functionSet

    def setFunctionSet(self, functionSet):
        self.functionSet = functionSet

    def getTerminalSet(self):
        return self.terminalSet

    def setTerminalSet(self, terminalSet):
        self.terminalSet = terminalSet

    def getArity(self):
        return self.arity

    def setArity(self, arity):
        self.arity = arity

    def getFunctionSetChoices(self):
        return self.functionSetChoices

    def setFunctionSetChoices(self, functionSetChoices):
        self.functionSetChoices = functionSetChoices