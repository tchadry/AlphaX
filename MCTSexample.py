class MCTSExampleGenerator:
    
    def __init__(self, problem, example_queue, iterations=1000):
        self.problem = problem
        self.root = MCTSNode(problem=self.problem)
        self.example_queue = example_queue
        self.iterations = iterations
    
    def solve(self):
        node = self.root
        end=node.is_leaf()
        while not end:
            node = self.mcts_search(node)
            self.generate_example(node.parent)
            end=node.is_leaf()
            
        path = node.get_tour()
        payoff = self.problem.tour_length(mcts_tour)
        return path, payoff

    def mcts_search(self, root):
        for _ in range(self.iterations):
            node = self.tree_policy(root)
            pay = node.simulate()
            node.backprop(pay)
        return root.best_child_score()

    def tree_policy(self, root):
        node = root
        end=node.is_leaf()
        while not end:
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.best_child_uct()
                end=node.is_leaf()
        return node
    
    def generate_example(self, node):
        
        # construct graph
        graph = node.construct_graph()
        
        # construct labels
        
        choice_probs = [(child.tour[-1], child.visits) for child in node.children]
        choice_probs = sorted(choice_probs, key = lambda c: c[1])
        choice_probs = [c[0] for c in choice_probs]
        choice_probs = torch.tensor(choice_probs).to(dtype=torch.float)
        
        #normalize by the sum 
        choice_probs = choice_probs / choice_probs.sum()
        
        #get the maximum probability
        choice = torch.argmax(choice_probs)
        
        pred_value = torch.tensor(node.avg_score)
        
        # add to training queue
        example = {
            "graph": graph,
            "choice_probs": choice_probs,
            "choice": choice,
            "pred_value": pred_value,
        }
        self.example_queue.put(example)
