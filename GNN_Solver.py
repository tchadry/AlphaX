class GNNSolver:
    
    def __init__(self, problem, model):
        self.problem = problem
        self.root = MCTSNode(problem=self.problem)
        
        #model will be either CNN of GNN 
        self.model = model
    
    def solve(self):
        node = self.root
        
        end = node.is_leaf() 
        while not end:
            node = node.best_remaining_policy(model=self.model)
            end = node.is_leaf() 
            
        path = node.get_complete_path()
        payoff = self.problem.tour_length(path)
        return path, payoff
