from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptance import MoveAcceptance
from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptanceClasses.AcceptAll import AcceptAll

from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniques import SelectionTechniques
from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniqueClasses.randomSelection import RandomSelection


class HyperHeuristic:
    def __init__(self, hyperHeuristic, selectionTechnique, moveAcceptance) -> None:
        self.hyperHeuristic = hyperHeuristic
        # self.numLowLevelHeuristicsToApply = hyperHeuristic.numLowLevelHeuristicsToApply
        self.numLowLevelHeuristicsToApply = 1
        self.selectionTechnique = self.createSelectionTechnique(selectionTechnique)
        self.moveAcceptance = self.createMoveAcceptance(moveAcceptance)

    def createSelectionTechnique(self, selectionTechnique):
        if selectionTechnique == "random":
            return RandomSelection(self.hyperHeuristic, self.numLowLevelHeuristicsToApply)
        # elif selectionTechnique == "choiceFunction":
        #     return ChoiceFunctionSelection()
        else:
            raise ValueError("Invalid selection technique")

    def createMoveAcceptance(self, moveAcceptance):
        if moveAcceptance == "acceptAll":
            return AcceptAll(self.hyperHeuristic)
        # elif moveAcceptance == "acceptBetter":
        #     return AcceptBetter()
        else:
            raise ValueError("Invalid move acceptance")

    def performSelectionAndMoveAcceptance(self):
        self.selectionTechnique.selection()
        self.moveAcceptance.accept()