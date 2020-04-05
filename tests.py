from Policy_GNN import GNN_Policy
import torch
from torch import nn
from tsp import TSP

training = []

policy_network = GNN_Policy()
optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()


# generate examples
for _ in range(25):
    tsp = TSP(20, 2)
    solver = MCTSExampleGenerator(tsp, train_queue)
    solver.solve()