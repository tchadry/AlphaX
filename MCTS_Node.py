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
        self.problem = problem

    def expand(self):
        remaining, visited = self.state.remaining[:], self.state.visited[:]
        next_action = random.choice(remaining)
        visited.append(next_action)
        remaining.remove(next_action)

        new_state = state.State(self.problem, remaining, visited)
        expanded_child = MCTS_Node(self, new_state, self.problem, next_action)

        self.children.append(expanded_child)

        return expanded_child