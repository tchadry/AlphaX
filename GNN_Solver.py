from Policy_GNN import GNN_Policy
from MCTS_Node import MCTS_Node
from GNNMCTS_Solver import GNN_MCTS_Solver
from GNNTrainer import GNN_Trainer
import torch
from torch import nn
from TSP import TSP
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class GNNSolver:

    def __init__(self, problem, model):
        self.problem = problem
        self.root = MCTS_Node(problem=self.problem)

        #model will be either CNN of GNN
        self.model = model

    def solve(self):
        node = self.root

        end = node.is_leaf()
        while not end:
            node = node.best_remaining_policy(model=self.model)
            end = node.is_leaf()

        path = node.get_path()
        payoff = self.problem.tour_length(path)
        return path, payoff
