import random
import copy
from TreeNode import TreeNode

class onePop:
    def __init__(self, chromosome) -> None:
        self.chromosome = chromosome
        self.rootNode = None
        self.output = []
        self.fitness = 0
        self.chromosomeIndex = 0

    def recursivlyBuildExpression(self, currentNode, currentDepth, maxDepth, expressions, expressionIndex):
        """
        Recursivly builds an expression from the population.
        """
        # check that chromosome index is valid 
        if(self.chromosomeIndex >= len(self.chromosome)):
            self.chromosomeIndex = 0

        # if current depth is equal to max depth set a terminal node
        if(currentDepth == maxDepth-1):
            currentNode.setVariable(expressions[-1][0])
            return

        # loop the amount of children to create and start creating them 
        numChildren = len(expressions[expressionIndex]) - 1
        for i in range(numChildren):
            # get new index for expression 
            expressionIndex = (self.stringToInt(self.chromosome[self.chromosomeIndex]) % len(expressions))
            currentNode.addChild(TreeNode(expressions[expressionIndex][0]))
            self.chromosomeIndex += 1

            # only recursively go down tree if current node is not a leaf
            if(currentNode.getVariable() != 'v'):
                self.recursivlyBuildExpression(currentNode.getChildAtIndex(i), currentDepth+1, maxDepth, expressions, expressionIndex)

        return

    def buildExpression(self, expressions, maxDepth):
        """
        Function to start recursive function 
        """
        chromosome = self.chromosome
        self.chromosomeIndex = 0

        # build first expression 
        expressionIndex = (self.stringToInt(chromosome[self.chromosomeIndex]) % len(expressions))
        self.rootNode = TreeNode(expressions[expressionIndex][0])
        self.chromosomeIndex += 1

        # check if the expression is a leaf otherwise recursivly build expression
        if(self.rootNode.getVariable() == 'v'):
            return
        self.recursivlyBuildExpression(self.rootNode, 0, maxDepth, expressions, expressionIndex)

    def buildTree(self, currentNode, dictonary):
        """
        Builds the tree using the dictionary
        """
        # check that chromosome index is valid 
        if(self.chromosomeIndex == len(self.chromosome)):
            self.chromosomeIndex = 0

        # if the current node is a leave
        if(currentNode.getVariable() == 'v'):
            dictonaryIndex = self.stringToInt(self.chromosome[self.chromosomeIndex]) % len(dictonary[currentNode.getVariable()])
            currentNode.setVariable(dictonary[currentNode.getVariable()][dictonaryIndex])
            self.chromosomeIndex += 1
            return 
        else:
            dictonaryIndex = self.stringToInt(self.chromosome[self.chromosomeIndex]) % len(dictonary[currentNode.getVariable()])
            currentNode.setVariable(dictonary[currentNode.getVariable()][dictonaryIndex])
            self.chromosomeIndex += 1
        
        # if the current node is not a leave
        for i in range(0, len(currentNode.getChildren())):
            self.buildTree(currentNode.getChildAtIndex(i), dictonary)

        return 

    def calculateOutput(self, inputs, outputs, dictionary, functionSet, functionSetChoices):
        """
        Function to calculate the output of the expression.
        """
        # build tree using dictionary
        self.buildTree(self.rootNode, dictionary)

        # list to hold each of the provided outputs to compare to the outputs for testing
        outputs = []

        # run for each inpup given and run it through the tree
        currentNode = self.rootNode
        for i in inputs:
            # loop through the tree until a terminal is found 
            while(currentNode.getVariable().isdigit() == False):
                # get the index of the function in current node and save the input value at that index
                functionIndex = functionSet.index(currentNode.getVariable())
                inputValue = i[functionIndex]

                # if the value is "?" then traverse down the edges 
                if(inputValue == "?"):
                    currentNode = currentNode.getChildAtIndex(0)
                else:
                    inputValue = float(inputValue)

                    # loop through the functionSetChoices and determin which range to pick
                    index = 0
                    for option in functionSetChoices[functionIndex]:
                        if(inputValue <= option):
                            break
                        index += 1
                    currentNode = currentNode.getChildAtIndex(index)

            # once terminal has been found save it to outputs and repeat for next input
            outputs.append(int(currentNode.getVariable()))
            currentNode = self.rootNode

        # save the output to the population
        self.output = outputs

    def stringToInt(self, string):
        """
        Converts the string to int
        """
        ans = 0
        for i in range(len(string)-1, -1, -1):
            if(string[(len(string)-1)-i] == "1"):
                ans += 2**i
        return ans

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
        newTree = onePop(copy.deepcopy(self.chromosome))    
        newTree.fitness = self.fitness
        newTree.output = self.output

        # create new root node and recursively copy the tree
        newTree.rootNode = TreeNode(self.rootNode.getVariable())
        self.recursivelyCopyTree(self.rootNode, newTree.rootNode)
        return newTree

    # Getters and Setters 
    def getChromosome(self):
        return self.chromosome

    def setChromosome(self, chromosome):
        self.chromosome = chromosome
    
    def setChromosomeAtIndex(self, index, chromosome):
        self.chromosome[index] = chromosome