# node class for the trees
class TreeNode:
    def __init__(self, data):
        self.value = 0
        self.variable = data
        self.children = []

    # getters and setters
    def getValue(self):
        return self.value

    def setValue(self, value):
        self.value = value

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