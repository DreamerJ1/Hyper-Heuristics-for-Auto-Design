class Tree():
    def __init__(self, maxDepth, functionSet, terminalSet):
        self.rootNode = None
        self.maxDepth = maxDepth
        self.functionSet = functionSet
        self.terminalSet = terminalSet
        self.fitness = 0
        self.output = []

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