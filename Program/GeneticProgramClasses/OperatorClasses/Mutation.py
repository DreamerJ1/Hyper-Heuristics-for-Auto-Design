import random
from Program.GeneticProgramClasses.Operators import Operators

class Mutation(Operators):
    """
    This class is used to mutate an individaul in the population
    """
    def __init__(self):
        pass

    def performOperation(self, pop):
        """
        Mutates the individual in the population
        """
        # generate random depth and create current node
        depth = random.randint(0, pop.maxDepth-1)
        currentNode = pop.rootNode

        # get random node at depth found 
        currentNode, depthFound = pop.getRandomNode(pop, depth)

        # give the current node a new variable and send it through to the recursive grow algorithm
        whichSet = random.randint(0, 1)
        if(whichSet == 0 and currentNode != pop.rootNode):
            currentNode.setVariable(pop.terminalSet[random.randint(0, len(pop.terminalSet)-1)])
            currentNode.setChildren([])
        else:
            currentNode.setVariable(pop.functionSet[random.randint(0, len(pop.functionSet)-1)])
            currentNode.setChildren([])
            pop.recursivelyCreateGrowTree(currentNode, depthFound)

        return pop
