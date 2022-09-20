class MoveAcceptance:
    def __init__(self, hyperHeuristic) -> None:
        self.hyperHeuristic = hyperHeuristic

    def accept(self):
        pass

    def calculateParetoDominance(self, oldParetoVector, newParetoVector, threshhold):
        """
        Calculate the pareto dominance of the new pareto vector compared to the old pareto vector using:

        Def inition: A vector u = (u1, ..., uk) is said to dominate another vector
        v = (v1, ..., vk) (denoted by u ≼ v) according to k objectives, if and only if,
        u is partially less than v, i.e., ∀ i ∈ {1, ..., k}: ui ≤ vi ∧ ∃ i ∈ {1, ..., k}: ui < vi

        Or in english, vector P is dominated by vector V if and only if all the objectives of P are less than or equal
        to the objectives of V and at least one objective of P is less than the corresponding objective of V.

        An adjustment is made to the formula to ensure that solutions that produce better accuracy are favoured as most of the time
        better accuracy solutions take longer to run and are more complex therefore the new equation is as follows: 

        (u1 < v1) V (∀ i ∈ {1, ..., k}: ui ≤ vi) ∧ (∃ i ∈ {1, ..., k}: ui < vi)
        """
        # check if there is a thershold to apply
        if(threshhold != None):
            # if accuracy is better 
            if(newParetoVector[0] < oldParetoVector[0]):
                return True, "better"

            # loop through each of the new pareto vectors objectives
            for i in range(len(newParetoVector)):
                # if any of the new pareto vectors is worse than the old ones return false
                if(newParetoVector[i] > oldParetoVector[i]):
                    return False, "worse"

            # loop through each of the new pareto vectors objectives
            for i in range(len(newParetoVector)):
                # if any of the new pareto vectors is better than the old ones return true
                if(newParetoVector[i] < oldParetoVector[i]):
                    return True, "better"

            # if they were all similar
            return True, "similar"
        else:
            # if accuracy is better than threshold
            if(newParetoVector[0] < (oldParetoVector[0] + threshhold[0])):
                return True, "better"

            # loop through each of the new pareto vectors objectives
            for i in range(len(newParetoVector)):
                # if any of the new pareto vectors is worse than the old ones return false
                if(newParetoVector[i] > (oldParetoVector[i] + threshhold[i])):
                    return False, "worse"

            # loop through each of the new pareto vectors objectives
            for i in range(len(newParetoVector)):
                # if any of the new pareto vectors is better than the old ones return true
                if(newParetoVector[i] < oldParetoVector[i] + threshhold[i]):
                    return True, "better"

            # if they were all similar
            return True, "similar"
        