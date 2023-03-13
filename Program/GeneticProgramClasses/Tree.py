import math
class Tree():
    def __init__(self, maxDepth, functionSet, terminalSet):
        self.rootNode = None
        self.maxDepth = maxDepth
        self.functionSet = functionSet
        self.terminalSet = terminalSet
        self.fitness = 0
        self.output = []

    # FUNCTIONS
    def countNumberOfFunctionalNodes(self, currentNode, count):
        """
        Count the number of functional nodes in the tree
        """    
        # break if no children
        if(currentNode.getChildren() == []):
            return count

        # loop through all children
        for i in range(len(currentNode.getChildren())):
            # check if child is a function
            if(currentNode.getChildAtIndex(i).getVariable() in self.functionSet):
                count = self.countNumberOfFunctionalNodes(currentNode.getChildAtIndex(i), count + 1)

        # return count if program is finished running 
        return count

    def countNumberOfTerminalNodes(self, currentNode, count):
        """
        Count the number of terminal nodes in the tree
        """    
        # break if no children
        if(currentNode.getChildren() == []):
            return count

        # loop through all children
        for i in range(len(currentNode.getChildren())):
            # check if child is a function
            if(currentNode.getChildAtIndex(i).getVariable() in self.terminalSet):
                count = self.countNumberOfTerminalNodes(currentNode.getChildAtIndex(i), count + 1)
            else:
                count = self.countNumberOfTerminalNodes(currentNode.getChildAtIndex(i), count)

        # return count if program is finished running 
        return count

    def calculateComplexity(self):
        """
        Calculate the complexity of the tree
        """
        f = len(self.functionSet)
        t = len(self.terminalSet)
        n_f = self.countNumberOfFunctionalNodes(self.rootNode, 1)
        n_t = self.countNumberOfTerminalNodes(self.rootNode, 0)
        return (n_f + n_t + (n_f * math.log(f, 2)) + (n_t * math.log(t, 2)))

    # GETTERS AND SETTERS
    def getRootNode(self):
        return self.rootNode

    def setRootNode(self, rootNode):
        self.rootNode = rootNode

    def getMaxDepth(self):
        return self.maxDepth

    def setMaxDepth(self, maxDepth):
        self.maxDepth = maxDepth

    def getFunctionSet(self):
        return self.functionSet

    def setFunctionSet(self, functionSet):
        self.functionSet = functionSet

    def getTerminalSet(self):
        return self.terminalSet

    def setTerminalSet(self, terminalSet):
        self.terminalSet = terminalSet

    def getFitness(self):
        return self.fitness

    def setFitness(self, fitness):
        self.fitness = fitness

    def getOutput(self):
        return self.output