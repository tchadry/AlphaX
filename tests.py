from Policy_GNN import GNN_Policy
from MCTS_Node import MCTS_Node
from MCTSexample import MCTSExample
from MCTS_solver import MCTS_Solver
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

training = []

policy_network = GNN_Policy()
optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()


# generate examples
for _ in range(25):
    tsp = TSP(20, 2)
    solver = MCTSExample(tsp, training)
    solver.solve()

# train
trainer = GNN_Trainer(policy_network, training)
trainer.train_all()
policy_network = trainer.model

# plot loss
plt.scatter(x=np.arange(len(trainer.losses)), y=trainer.losses, marker='.')
plt.show()
