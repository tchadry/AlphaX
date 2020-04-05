import random
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader

class MCTS_Node:
    def __init__(self, problem=None, parent=None, action=None, path=[0], remaining=None):
        self.parent = parent
        self.children = []
        self.graph = None
        self.action = action
        self.visit_count = 0
        self.score = 0
        self.avg_score = 0
        self.problem = problem
        self.path = path

        if remaining is None:
            self.remaining = [i for i in range(self.problem.n)]
        else:
            self.remaining = remaining

    def expand(self):
        # expand on current node with a random possible action
        remaining = self.remaining[:]
        path = self.path[:]
        next_action = random.choice(remaining)
        path.append(next_action)
        remaining.remove(next_action)

        expanded_child = MCTS_Node(
            parent=self,
            problem=self.problem,
            action=next_action, 
            path=path, 
            remaining=remaining
        )

        self.children.append(expanded_child)

        return expanded_child


    def perform_action(self, action):
        # get child with action, if action was never performed, expand node
        for child in self.children:
            if child.action == action:
                return child
        child = self.expand()
        return child
            
    def backpropagate(self, reward):
        # add the reward and go up on the tree
        self.visit_count += 1
        self.score += reward
        self.avg_score = self.score / self.visit_count
        if self.parent:
            self.parent.backprop(reward)

    def simulate(self):
        # TODO: instead o naively generating the path, actually do a loop creating every state until it is terminal
        # otherwise it won't solve problems where the graphs are not fully connected
        random.shuffle(self.remaining)
        simulated_expansion = self.path + self.remaining + [0]
        return self.problem.payoff(simulated_expansion)

    def has_children(self):
        return len(self.children) != 0

    def is_leaf(self):
        return len(self.path) == self.problem.n
    
    def is_fully_expanded(self):
        return len(self.remaining) == len(self.children)

    def get_complete_path(self):
        return self.path + [self.path[0]]

    def UCT(self):
        epsilon = 0.5
        return self.avg_score + 2 * epsilon * np.sqrt(2 * np.log(self.parent.visit_count)/self.visit_count)

    def select_child(self):
        return max(self.children, key=lambda child: child.UCT())

    def create_graph(self):
        if self.graph:
            return self.graph

        nodes = torch.tensor(self.problem.points).to(dtype=torch.float)
        edges = torch.zeros((2, len(self.path)-1), dtype=torch.long)

        for i in range(len(self.path)-1):
            edges[0, i] = self.path[i]
            edges[1, i] = self.path[i+1]

        choices = torch.zeros(self.problem.n, dtype=torch.uint8)
        choices[self.remaining] = 1
        
        x = torch.cat([nodes, choices.unsqueeze(-1).to(dtype=torch.float)], dim=-1)

        self.graph = Data(x=x, pos=nodes, edge_index=edges, y=choices)
        
        return self.graph