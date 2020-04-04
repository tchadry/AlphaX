class State:
    def __init__(self, problem, remaining=None, path=[0]):
        self.path = path
        self.problem = problem
        
        if remaining is None:
            self.remaining = [i for i in range(problem.n)]
        else:
            self.remaining = remaining