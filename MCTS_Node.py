class MCTS_Node:
    def __init__(self, parents, state, problem, action):
        self.parents = parents
        self.state = state
        self.children = []
        self.graph = None
        self.action = action
        self.visit_count = 0