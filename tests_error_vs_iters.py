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
import seaborn as sns



solve_iterations = 25
nodes = 8
test_iterations = 100

errors = []

#iterations = list(range(25))
iterations = [500, 1000, 1500, 2000,2500,3000,3500,4000,4500,6000] #, 30, 35, 45, 50]

for iters in iterations:

    training = []

    policy_network = GNN_Policy()
    optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    # generate examples
    for i in range(solve_iterations):
        print(f"Iterating {i}/{solve_iterations}")
        tsp = TSP(nodes, 2)
        solver = MCTSExample(tsp, training, iterations=iters)
        solver.solve()
    #print(training)
    # train
    trainer = GNN_Trainer(policy_network, training)
    trainer.train_all()
    policy_network = trainer.model
    #print(trainer.losses)
    # plot loss
    #plt.scatter(x=np.arange(len(trainer.losses)), y=trainer.losses, marker='.')
    #plt.show()

    # test choices of policy network
    #print("start testing")
    results = []

    error = 0

    for _ in range(test_iterations):
        tsp = TSP(nodes, 2)
        solver = GNNSolver(tsp, policy_network)

        result = solver.solve()

        results.append(result)

        optimal_length = tsp.get_optimal_length()
        if (result[1]/optimal_length) <1.5: 
            error += 1

    error /= test_iterations
    errors.append(error)

    # print(results)
    # print("DONE!")

plt.scatter(x=iterations, y=errors, marker='.')
plt.show()
#sns.lineplot(x=iterations, y=errors)
