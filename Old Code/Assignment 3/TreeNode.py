# node class for the trees
class TreeNode:
    def __init__(self, data):
        self.variable = data
        self.children = []

    # getters and setters
    def getVariable(self):
        return self.variable

    def setVariable(self, variable):
        self.variable = variable

    def getChildren(self):
        return self.children

    def setChildren(self, children):
        self.children = children

    def addChild(self, node):
        self.children.append(node)

    def deleteChild(self, node):
        self.children.remove(node)

    def getChildAtIndex(self, index):
        return self.children[index]