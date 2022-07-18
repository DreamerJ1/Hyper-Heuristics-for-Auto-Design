import random
import uuid
from treelib import Tree as TreeLib

from Program.GeneticProgramClasses.Tree import Tree 
from Program.GeneticProgramClasses.TreeClasses.DecisionTree import DecisionTree

class GeneticProgram:
    def __init__(self, inisialGenerationOptions: dict, data, dataType) -> None:
        self.parameters = inisialGenerationOptions
        self.terminalSet = []
        self.functionSet = []
        self.functionSetChoices = []
        self.arity = []
        self.inputHandling(data, dataType)

    def inputHandling(self, data, dataType):
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

    def createPopulation(self, popSize, maxDepth, functionSet, terminalSet, arity, functionSetChoices, method):
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
        pass

    def SelectionMethodSelection(self):
        """
        Selects the selection method to be used
        """
        pass

    def operatorSelection(self):
        """
        Selects the operator to be used
        """
        pass

    def preformGeneticOperators(self, population, output):
        """
        Preform the genetic operators for the genetic program
        """
        # create new population 
        newPopulation = []

        return population

    def changeTreeAndPrint(self, currentNode, id, isRoot, tree):
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


    def createTree(self, pop):
        """ 
        Creates a tree using the given population
        """
        tree = TreeLib()
        self.changeTreeAndPrint(pop.rootNode, str(uuid.uuid4()), True, tree)
        tree.show()

    def runGeneticProgram(self):
        """
        The main function of the genetic program
        """
        # store the parameters of the gp in a text file 
        with open("Storage\parameterStorage.txt", 'a') as f: 
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

        self.createTree(population[0])

        return self