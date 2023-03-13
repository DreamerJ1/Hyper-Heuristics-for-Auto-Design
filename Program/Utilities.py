class Utilities():
    def __init__(self) -> None:
        pass

    def checkIfNumber(self, value):
        try:
            float(value)
            return float(value)
        except:
            return value