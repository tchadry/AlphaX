class TSP_State:
    def __init__(self, tsp_problem, remaining=None, visited=[]):
        self.visited = visited
        self.tsp_problem = tsp_problem
        
        if remaining is None:
            self.remaining = [i for i in range(tsp_problem.n)]
        else:
            self.remaining = remaining