class State:
    def __init__(self, problem, remaining=None, visited=[0]):
        self.visited = visited
        self.problem = problem
        
        if remaining is None:
            self.remaining = [i for i in range(problem.n)]
        else:
            self.remaining = remaining