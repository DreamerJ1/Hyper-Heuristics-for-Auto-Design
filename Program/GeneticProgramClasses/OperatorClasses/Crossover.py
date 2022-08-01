import random

from Program.GeneticProgramClasses.Operators import Operators

class Crossover(Operators):
    def __init__(self) -> None:
        pass

    def performOperation(self, popOne, popTwo):
        """
        Perform crossover on the two elements of the population
        """
        # get depthOne and depthTwo
        depthOne = random.randint(0, popOne.maxDepth-1)
        depthTwo = random.randint(0, popTwo.maxDepth-1)

        # get nodes just above each depth
        nodeOne, depthOneFound = popOne.getRandomNode(popOne, depthOne-1)
        nodeTwo, depthTwoFound = popTwo.getRandomNode(popTwo, depthTwo-1)

        # pick directions that are functional nodes 
        canPickFromForPopOne = []
        canPickFromForPopTwo = []
        for i in range(nodeOne.getChildren().__len__()):
            if(nodeOne.getChildAtIndex(i).getVariable() in popOne.functionSet):
                canPickFromForPopOne.append(i)
        for i in range(nodeTwo.getChildren().__len__()):
            if(nodeTwo.getChildAtIndex(i).getVariable() in popTwo.functionSet):
                canPickFromForPopTwo.append(i)

        # pick one of the valid directions 
        if(canPickFromForPopOne.__len__() > 0):
            popOneDir = canPickFromForPopOne[random.randint(0, canPickFromForPopOne.__len__()-1)]
        else:
            popOneDir = random.randint(0, popOne.arity[popOne.functionSet.index(nodeOne.getVariable())]-1)
        if(canPickFromForPopTwo.__len__() > 0):
            popTwoDir = canPickFromForPopTwo[random.randint(0, canPickFromForPopTwo.__len__()-1)]
        else:
            popTwoDir = random.randint(0, popTwo.arity[popTwo.functionSet.index(nodeTwo.getVariable())]-1)

        # swap the two nodes
        popOneChildHolder = nodeOne.getChildren()
        popTwoChildHolder = nodeTwo.getChildren()
        temp = popOneChildHolder[popOneDir]
        popOneChildHolder[popOneDir] = popTwoChildHolder[popTwoDir]
        popTwoChildHolder[popTwoDir] = temp

        # insert the children back to node
        nodeOne.setChildren(popOneChildHolder)
        nodeTwo.setChildren(popTwoChildHolder)