from Program.HyperHeuristicClasses.MoveAcceptance.MoveAcceptance import MoveAcceptance


class AcceptAll(MoveAcceptance):
    def __init__(self, hyperHeuristic) -> None:
        super().__init__(hyperHeuristic)

    def accept(self):
        return True