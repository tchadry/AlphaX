import random
import state

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
        remaining = self.state.remaining[:]
        visited = self.state.visited[:]
        next_action = random.choice(remaining)
        visited.append(next_action)
        remaining.remove(next_action)

        new_state = state.State(self.problem, remaining, visited)
        expanded_child = MCTS_Node(self, new_state, self.problem, next_action)

        self.children.append(expanded_child)

        return expanded_child


    def perform_action(self, action):
        for child in self.children:
            if child.action == action:
                return child
        child = self.expand()
        return child
            
    def backpropagate(self, reward):
        self.visit_count += 1
        self.score += reward
        self.avg_score = self.score / self.visit_count
        if self.parent:
            self.parent.backprop(reward)
