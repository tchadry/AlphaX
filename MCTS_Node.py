import random
import state
import numpy as np
import torch

class MCTS_Node:
    def __init__(self, parent, state, problem, action):
        self.parent = parent
        self.state = state
        self.children = []
        self.graph = None
        self.action = action
        self.visit_count = 0
        self.score = 0
        self.avg_score = 0
        self.problem = problem

    def expand(self):
        # expand on current node with a random possible action
        remaining = self.state.remaining[:]
        path = self.state.path[:]
        next_action = random.choice(remaining)
        path.append(next_action)
        remaining.remove(next_action)

        new_state = state.State(self.problem, remaining, path)
        expanded_child = MCTS_Node(self, new_state, self.problem, next_action)

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
        random.shuffle(self.state.remaining)
        simulated_expansion = self.state.path + self.state.remaining + [0]
        return self.problem.payoff(simulated_expansion)

    def has_children(self):
        return len(self.children) != 0

    def is_leaf(self):
        return len(self.state.path) == self.problem.n
    
    def is_fully_expanded(self):
        return len(self.state.remaining) == len(self.children)

    def get_complete_path(self):
        return self.state.path + [self.state.path[0]]

    def UCT(self):
        epsilon = 0.5
        return self.avg_score + 2 * epsilon * np.sqrt(2 * np.log(self.parent.visit_count)/self.visit_count)

    def select_child(self):
        return max(self.children, key=lambda child: child.UCT())

    def predict_next_child(self, model):
        pass
