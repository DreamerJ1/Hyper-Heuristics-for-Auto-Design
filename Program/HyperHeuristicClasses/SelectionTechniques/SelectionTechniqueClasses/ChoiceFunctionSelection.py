from Program.HyperHeuristicClasses.SelectionTechniques.SelectionTechniques import SelectionTechniques


class ChoiceFunctionSelection(SelectionTechniques):
    def __init__(self, hyperHeuristic, numToSelect) -> None:
        super().__init__(hyperHeuristic, numToSelect)