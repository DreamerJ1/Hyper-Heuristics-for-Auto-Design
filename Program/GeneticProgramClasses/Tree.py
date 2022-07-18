class Tree():
    def __init__(self, maxDepth, functionSet, terminalSet):
        self.rootNode = None
        self.maxDepth = maxDepth
        self.functionSet = functionSet
        self.terminalSet = terminalSet
        self.fitness = 0
        self.output = []