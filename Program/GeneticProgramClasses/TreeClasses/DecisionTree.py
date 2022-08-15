import random

from Program.GeneticProgramClasses.Tree import Tree
from Program.GeneticProgramClasses.TreeClasses.DTNode import DTNode

class DecisionTree(Tree):
    def __init__(self, maxDepth, functionSet, terminalSet, arity, functionSetChoices):
        super().__init__(maxDepth, functionSet, terminalSet)
        self.arity = arity
        self.functionSetChoices = functionSetChoices

    # ENTIRE TREE FUNCTIONS

    def printTree(self, currentNode, currentDepth):
        print(currentNode.getVariable(), currentDepth)

        for i in range(0, len(currentNode.getChildren())):
            self.printTree(currentNode.getChildAtIndex(i), currentDepth+1)

    def checkTerminals(self, node):
        """
        Checks if the children of a node are only terminals
        """
        for i in range(node.getChildren().__len__()):
            if(node.getChildren()[i].getChildren().__len__() == 0):
                return True
        return False

    def getRandomNode(self, tree, maxDepth):
        """
        Gets a random node from the tree
        """
        currentDepth = 0
        currentNode = tree.rootNode
        parentNode = currentNode
        parentsParentNode = currentNode
        while(currentDepth <= maxDepth):
            index = self.functionSet.index(currentNode.getVariable())
            parentsParentNode = parentNode
            parentNode = currentNode
            currentNode = currentNode.getChildAtIndex(random.randint(0, self.arity[index]-1))
            currentDepth += 1

            # check if we hit a terminal node since we cant go down farther
            if(currentNode.getChildren() == [] or self.checkTerminals(currentNode)):
                return parentsParentNode, currentDepth-2

        return currentNode, currentDepth-1

    def getRandomNodeMutation(self, tree, maxDepth):
        """
        Gets a random node from the tree
        """
        currentDepth = 0
        currentNode = tree.rootNode
        parentNode = currentNode
        parentsParentNode = currentNode
        while(currentDepth <= maxDepth):
            index = self.functionSet.index(currentNode.getVariable())
            parentsParentNode = parentNode
            parentNode = currentNode
            currentNode = currentNode.getChildAtIndex(random.randint(0, self.arity[index]-1))
            currentDepth += 1

            # check if we hit a terminal node since we cant go down farther
            if(currentNode.getChildren() == [] or self.checkTerminals(currentNode)):
                return currentNode, currentDepth

        return currentNode, currentDepth

    def recursivelyCopyTree(self, currentNodeToCopyFrom, currentNodeToCopyTo):
        """
        Recursively copies the node from the currentNodeToCopyFrom to the currentNodeToCopyTo
        """
        if(currentNodeToCopyFrom.getVariable().isdigit()):
            currentNodeToCopyTo.setVariable(currentNodeToCopyFrom.getVariable())
            return
        else:
            currentNodeToCopyTo.setVariable(currentNodeToCopyFrom.getVariable())
            for i in range(0, len(currentNodeToCopyFrom.getChildren())):
                currentNodeToCopyTo.addChild(DTNode(currentNodeToCopyFrom.getChildAtIndex(i).getVariable()))
                self.recursivelyCopyTree(currentNodeToCopyFrom.getChildAtIndex(i), currentNodeToCopyTo.getChildAtIndex(i))


    def deepCopy(self):
        """
        Returns a deep copy of the current tree
        """
        # create new poptree
        newTree = DecisionTree(self.maxDepth, self.functionSet, self.terminalSet, self.arity, self.functionSetChoices)
        newTree.fitness = self.fitness
        newTree.output = self.output

        # create new root node and recursively copy the tree
        newTree.rootNode = DTNode(self.rootNode.getVariable())
        self.recursivelyCopyTree(self.rootNode, newTree.rootNode)
        return newTree

    # FUNCTIONS SPECIFIC TO GROW TREES

    def recursivelyCreateGrowTree(self, currentNode, currentDepth):
        # break for recursion
        if(currentDepth == self.maxDepth):
            return

        # Check if both the terminalSet and functionSet can be used
        nextDepth = currentDepth+1
        if(nextDepth == self.maxDepth):
            # randomly select one of the terminals sets and assign it to the variable of the current node
            currentNode.setVariable(
                self.terminalSet[random.randint(0, len(self.terminalSet)-1)])
            return
        else:
            # randomly select one of the function sets and assign it to the variable of the current node
            functionSetElementIndex = self.functionSet.index(
                currentNode.getVariable())

            # loop through the functionset's choices and create new node for each choice
            for i in range(0, self.arity[functionSetElementIndex]):
                # check if we use a function or terminal set for next node
                decide = random.randint(0, 1)
                if(decide == 0):
                    # create a terminal node
                    currentNode.addChild(DTNode(self.terminalSet[random.randint(
                        0, len(self.terminalSet)-1)]))
                else:
                    # create a function node
                    currentNode.addChild(
                        DTNode(self.functionSet[random.randint(0, len(self.functionSet)-1)]))
                    self.recursivelyCreateGrowTree(
                        currentNode.getChildAtIndex(i), nextDepth)

    def growGeneration(self):
        # create the root node
        functionElement = self.functionSet[random.randint(
            0, len(self.functionSet)-1)]
        self.rootNode = DTNode(functionElement)

        # fill tree according to grow method
        self.recursivelyCreateGrowTree(self.rootNode, 0)

        return

    # FUNCTIONS SPECIFIC TO FILL TREES

    def recursivelyCreateFullTree(self, currentNode, currentDepth):
        # break for recursion
        if(currentDepth == self.maxDepth):
            return

        # Check if both the terminalSet and functionSet can be used
        nextDepth = currentDepth+1
        if(nextDepth == self.maxDepth):
            # randomly select one of the terminals sets and assign it to the variable of the current node
            currentNode.setVariable(
                self.terminalSet[random.randint(0, len(self.terminalSet)-1)])
            return
        else:
            # randomly select one of the function sets and assign it to the variable of the current node
            functionSetElementIndex = self.functionSet.index(
                currentNode.getVariable())

            # loop through the functionset's choices and create new node for each choice
            for i in range(0, self.arity[functionSetElementIndex]):
                # create a function node
                currentNode.addChild(
                    DTNode(self.functionSet[random.randint(0, len(self.functionSet)-1)]))
                self.recursivelyCreateGrowTree(
                    currentNode.getChildAtIndex(i), nextDepth)

    def fullGeneration(self):
        # create the root node
        functionElement = self.functionSet[random.randint(0, len(self.functionSet)-1)]
        self.rootNode = DTNode(functionElement)

        # call the recursive function to create the tree
        self.recursivelyCreateFullTree(self.rootNode, 0)

        return

    # CALCULATING OUTPUT FOR INPUT
    def calculateOutput(self, inputs):
        # list to hold each of the provided outputs to compare to the outputs for testing
        outputs = []

        # run for each inpup given and run it through the tree
        currentNode = self.rootNode
        for i in inputs:
            # loop through the tree until a terminal is found 
            while(len(currentNode.getChildren()) > 0):
                # get the index of the function in current node and save the input value at that index
                functionIndex = self.functionSet.index(currentNode.getVariable())
                inputValue = i[functionIndex]

                # if the value is "?" then traverse down the edges 
                if(inputValue == "?"):
                    currentNode = currentNode.getChildAtIndex(0)
                else:
                    # loop through the functionSetChoices and determin which range to pick
                    index = self.functionSetChoices[functionIndex].index(inputValue)
                    currentNode = currentNode.getChildAtIndex(index)

            # once terminal has been found save it to outputs and repeat for next input
            outputs.append(currentNode.getVariable())
            currentNode = self.rootNode

        # save the output to the population
        self.output = outputs

    # recursively run through the tree and count amount of nodes 
    def countNodes(self, currentNode):
        if(currentNode.getVariable().isdigit()):
            self.nLeaves += 1
            if(int(currentNode.getVariable()) > 0):
                self.nPositive += 1
            else:
                self.nNegative += 1
        else:
            self.nFunctionalNodes += 1
            for i in range(0, len(currentNode.getChildren())):
                self.countNodes(currentNode.getChildAtIndex(i))

    # get the amount of functional and leave nodes in this tree
    def getInfo(self):
        # reset info 
        self.nFunctionalNodes = 0
        self.nLeaves = 0
        currentNode = self.rootNode
        self.countNodes(currentNode)
        return self.nFunctionalNodes, self.nLeaves

    # GETTERS AND SETTERS
    def getArity(self):
        return self.arity

    def setArity(self, arity):
        self.arity = arity

    def functionSetChoices(self):
        return self.functionSetChoices

    def setFunctionSetChoices(self, functionSetChoices):
        self.functionSetChoices = functionSetChoices