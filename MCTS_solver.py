hello

MCTSSolver:
    
#given the game, and number of iterations 
    def _init_(self, tsp, iterations=1000):
        self.tsp = tsp
        self.root_node = MCTSNode(tsp=self.tsp)
        self.iterations = iterations

#to solve, 
    def solve(self):
	# we start at the node 
        node = self.root_node

	#while not at the end 
        while not node.is_leaf():
		#do the MCTS search on the node 
		#MCTS search: the selection, expansion, simulation and backprop
            node = self.mcts_search(node)

	#from the node, we get the tour 
        mcts_tour = node.get_tour()
        mcts_payoff = self.tsp.tour_length(mcts_tour)

	
        return mcts_tour, mcts_payoff
	
	#search: 
    def mcts_search(self, start_node):
        for _ in range(self.iterations):
            node = self.tree_policy(start_node)
            pay = node.simulate()
            node.backprop(pay)
        return start_node.best_child_score()

    def tree_policy(self, start_node):
        node = start_node
        while not node.is_leaf():
            if not node.is_fully_expanded():
                return node.expand()
            else:
                node = node.select_child()
        return node
