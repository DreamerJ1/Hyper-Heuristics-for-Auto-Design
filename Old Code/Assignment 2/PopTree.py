import random
from TreeNode import TreeNode


class PopTree:
    def __init__(self, maxDepth, functionSet, terminalSet, arity, functionSetChoices):
        self.rootNode = None
        self.maxDepth = maxDepth
        self.functionSet = functionSet
        self.terminalSet = terminalSet
        self.arity = arity
        self.functionSetChoices = functionSetChoices
        self.fitness = 0
        self.output = []
        self.nFunctionalNodes = 0
        self.nLeaves = 0
        self.nPositive = 0
        self.nNegative = 0

    # ENTIRE TREE FUNCTIONS

    def printTree(self, currentNode, currentDepth):
        print(currentNode.getVariable(), currentDepth)

        for i in range(0, len(currentNode.getChildren())):
            self.printTree(currentNode.getChildAtIndex(i), currentDepth+1)

        return

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
                currentNodeToCopyTo.addChild(TreeNode(currentNodeToCopyFrom.getChildAtIndex(i).getVariable()))
                self.recursivelyCopyTree(currentNodeToCopyFrom.getChildAtIndex(i), currentNodeToCopyTo.getChildAtIndex(i))


    def deepCopy(self):
        """
        Returns a deep copy of the current tree
        """
        # create new poptree
        newTree = PopTree(self.maxDepth, self.functionSet, self.terminalSet, self.arity, self.functionSetChoices)
        newTree.fitness = self.fitness
        newTree.output = self.output

        # create new root node and recursively copy the tree
        newTree.rootNode = TreeNode(self.rootNode.getVariable())
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
                    currentNode.addChild(TreeNode(self.terminalSet[random.randint(
                        0, len(self.terminalSet)-1)]))
                else:
                    # create a function node
                    currentNode.addChild(
                        TreeNode(self.functionSet[random.randint(0, len(self.functionSet)-1)]))
                    self.recursivelyCreateGrowTree(
                        currentNode.getChildAtIndex(i), nextDepth)

    def growGeneration(self):
        # create the root node
        functionElement = self.functionSet[random.randint(
            0, len(self.functionSet)-1)]
        self.rootNode = TreeNode(functionElement)

        # fill tree according to grow method
        self.recursivelyCreateGrowTree(self.rootNode, 0)

        return

    # FUNCTIONS SPECIFIC TO FILL TREES

    def recursivelyCreateFullTree(self, currentNode, currentDepth):
        # break for recursion
        if(currentDepth == self.maxDepth):
            return

    def fullGeneration(self):
        # create the root node
        functionElement = self.functionSet[random.randint(
            0, len(self.functionSet)-1)]
        self.rootNode = TreeNode(functionElement)

        # call the recursive function to create the tree
        self.recursivelyCreateFullTree(self.rootNode, 0)

        return

    # CALCULATING OUTPUT FOR INPUT

    # check which type of comparison we are doing
    def whichTraversal(self, variable):
        if(variable == "age" or variable == "trestbps" or variable == "thalach" or variable == "oldpeak"):
            return False
        return True

    def calculateOutput(self, inputs):
        # list to hold each of the provided outputs to compare to the outputs for testing
        outputs = []

        # run for each inpup given and run it through the tree
        currentNode = self.rootNode
        for i in inputs:
            # loop through the tree until a terminal is found 
            while(currentNode.getVariable().isdigit() == False):
                # get the index of the function in current node and save the input value at that index
                functionIndex = self.functionSet.index(currentNode.getVariable())
                inputValue = i[functionIndex]

                # if the value is "?" then traverse down the edges 
                if(inputValue == "?"):
                    currentNode = currentNode.getChildAtIndex(0)
                else:
                    inputValue = float(inputValue)

                    # loop through the functionSetChoices and determin which range to pick
                    index = 0
                    for option in self.functionSetChoices[functionIndex]:
                        if(inputValue <= option):
                            break
                        index += 1
                    currentNode = currentNode.getChildAtIndex(index)

            # once terminal has been found save it to outputs and repeat for next input
            outputs.append(int(currentNode.getVariable()))
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