from Policy_GNN import GNN_Policy
from MCTS_Node import MCTS_Node
from MCTSGNNexample import MCTSGNNexample
from GNN_Solver import GNNSolver
from GNNMCTS_Solver import GNN_MCTS_Solver
from GNNTrainer import GNN_Trainer
import torch
from torch import nn
from TSP import TSP
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class MCTS_Solver:

    #to initialize the solver, we will test different number of iterations; and game also has to be given
    def _init_(self, problem, iterations=1000):
        self.problem = problem
        self.root = MCTS_Node(problem=self.problem)
        self.iterations = iterations

    #to solve, we are going to simulate  MCTS, and return the path and payoff
    def solve(self):
    # we start at the node
        node = self.root

        end = node.is_leaf()
        #while not at the end
        while not end:
        #do the MCTS search on the node
        #MCTS search: the selection, expansion, simulation and backpropagate
            node = self.mcts_search(node)
        end = node.is_leaf()

        #from the end, we get the path and payoff until there
        path = node.get_complete_graph()
        payoff = self.problem.tour_length(path)


        return path, payoff

    #search:
    def mcts_search(self, root):
        for _ in range(self.iterations):
        #tree policy does selection and expansion
            node = self.tree_policy(root)
            pay = node.simulate()
            node.backpropagate(pay)
        return root.select_child_score()

    def tree_policy(self, root):
        node = root
        end = node.is_leaf()

    #iterate until we get to leaf node
        while not end:
        #if this node hasnt been expanded fully, we do expansion phase of MCTS
            if not node.is_fully_expanded():
                return node.expand()
            else:
                #if it has been, we select the best child
                node = node.select_child()
        end=node.is_leaf()

        return node
