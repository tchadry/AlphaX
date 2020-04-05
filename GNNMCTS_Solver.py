class GNN_MCTS_Solver:
    #we will solve the MCTS with the Neural network
    def __init__(self, problem, model, iterations=50):
        self.problem = problem
        self.root = MCTSNode(problem=self.problem)
        self.iterations = iterations
        self.model = model
    
    def solve(self):
        node = self.root
        end = node.is_leaf()
        
        #iterate until leaf node 
        while not end:
            node = self.mcts_search(node)
            end=node.is_leaf()
            
        path = node.get_tour()
        payoff = self.problem.tour_length(path)
        return path, payoff

    def mcts_search(self, root):
        for _ in range(self.iterations):
            node = self.tree_policy(root)
            pay = node.simulate()
            node.backprop(pay)
        return start_node.best_child_score()

    def tree_policy(self, root):
        node = root
        end=node.is_leaf()
        while not end:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.select_child_policy(self.model)
                end=node.is_leaf()
        return node
