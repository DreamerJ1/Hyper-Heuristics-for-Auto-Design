from Program.HyperHeuristic.SelectionTechniques.Techniques import Techniques
from Program.HyperHeuristic.SelectionTechniques.Techniques.randomSelection import RandomSelection


class HyperHeuristic:
    def __init__(self, hyperHeuristic, selectionTechnique, moveAcceptance) -> None:
        self.hyperHeuristic = hyperHeuristic
        self.selectionTechnique = self.createSelectionTechnique(selectionTechnique)
        self.moveAcceptance = self.createMoveAcceptance(moveAcceptance)

    def createSelectionTechnique(self, selectionTechnique):
        if selectionTechnique == "random":
            return RandomSelection(self.hyperHeuristic)
        # elif selectionTechnique == "choiceFunction":
        #     return ChoiceFunctionSelection()
        else:
            raise ValueError("Invalid selection technique")
