import numpy as np


# MCTS Node
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior_prob = 0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, actions, priors):
        for action, prior in zip(actions, priors):
            if action not in self.children:
                self.children[action] = MCTSNode(self.state.next_state(action), parent=self)
                self.children[action].prior_prob = prior

    def select_child(self, c_puct=1.0):
        return max(
            self.children.items(),
            key=lambda item: item[1].value + c_puct * item[1].prior_prob * np.sqrt(self.visits) / (1 + item[1].visits)
        )

    def backpropagate(self, value):
        self.value_sum += value
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(-value)

# MCTS Algorithm
class MCTS:
    def __init__(self, policy_network, value_network, action_dim):
        self.policy_network = policy_network
        self.value_network = value_network
        self.action_dim = action_dim

    def search(self, root, num_simulations=100, c_puct=1.0):
        for _ in range(num_simulations):
            node = root
            # Selection
            while node.is_expanded():
                action, node = node.select_child(c_puct)

            # Expansion
            state_vector = node.state.get_state_vector()
            priors = self.policy_network(state_vector[None, :]).numpy()[0]
            node.expand(range(self.action_dim), priors)

            # Evaluation
            value = self.value_network(state_vector[None, :]).numpy()[0]
            node.backpropagate(value)

        # Extract search results
        actions = list(root.children.keys())
        visits = [child.visits for child in root.children.values()]
        return actions, visits