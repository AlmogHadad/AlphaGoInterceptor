import numpy as np
from collections import defaultdict
import math


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state        # Current positions of objects (custom format)
        self.parent = parent      # Parent node
        self.action = action      # Action that led to this state
        self.children = []        # Child nodes
        self.visits = 0           # Visit count
        self.value_sum = 0        # Sum of value estimates

    def add_child(self, child_state, action):
        child = Node(child_state, parent=self, action=action)
        self.children.append(child)
        return child

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration_weight=1.0):
        ucb_values = [
            (child.value_sum / (child.visits + 1e-6)) + exploration_weight * np.sqrt(
                np.log(self.visits + 1) / (child.visits + 1e-6)
            )
            for child in self.children
        ]
        return self.children[np.argmax(ucb_values)]


class MCTS:
    def __init__(self, policy_network, value_network, simulations=100):
        """
        Initialize MCTS.
        Args:
            policy_network: Neural network predicting move probabilities.
            value_network: Neural network predicting state value.
            simulations: Number of simulations to run per move.
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.simulations = simulations
        self.tree = defaultdict(lambda: {"N": 0, "W": 0, "Q": 0, "P": None, "children": {}})

    def uct(self, parent, move):
        """
        Upper Confidence Bound for Trees (UCT).
        Args:
            parent: Node representing the parent state.
            move: Move being evaluated.
        Returns:
            UCT score.
        """
        child = parent["children"].get(move)
        if child is None or child["N"] == 0:
            return float('inf')  # Prioritize unvisited nodes
        return child["Q"] + math.sqrt(2 * math.log(parent["N"] + 1) / child["N"])

    def select(self, node):
        """
        Select the child node with the highest UCT score.
        Args:
            node: Current node.
        Returns:
            The best move and the child node.
        """
        best_move, best_child = None, None
        max_uct = -float('inf')
        for move, child in node["children"].items():
            uct_score = self.uct(node, move)
            if uct_score > max_uct:
                max_uct = uct_score
                best_move, best_child = move, child
        return best_move, best_child

    def expand(self, node, board):
        """
        Expand the node by adding all possible children.
        Args:
            node: Current node.
            board: Current board state.
        """
        valid_moves = self.get_valid_moves(board)
        if not valid_moves:
            return
        policy_probs = self.policy_network.predict(board[np.newaxis, ..., np.newaxis], verbose=0)[0]
        for move in valid_moves:
            start_x, start_y, end_x, end_y = move
            # Convert the move to the correct index for policy_probs
            move_index = start_x * 10 + start_y  # The index in the policy_probs corresponds to the starting position

            # Ensure the move hasn't already been added to the children
            if move not in node["children"]:
                node["children"][move] = {
                    "N": 0,  # Visits count
                    "W": 0,  # Total reward
                    "Q": 0,  # Action value
                    "P": policy_probs[move_index],  # Policy probability for this move (based on the starting position)
                    "children": {}
                }


    def simulate(self, board):
        """
        Perform a single MCTS simulation.
        Args:
            board: Current board state.
        """
        path = []
        node = self.tree[tuple(board.flatten())]

        # Traverse the tree to a leaf node
        while node["children"]:
            move, child = self.select(node)
            path.append((node, move))
            node = child
            board = self.apply_move(board, move)

        # Expand leaf node
        if not self.is_terminal(board):
            self.expand(node, board)

        # Simulate a random outcome or use value network
        if self.is_terminal(board):
            value = self.evaluate_terminal(board)
        else:
            value = self.value_network.predict(board[np.newaxis, ..., np.newaxis], verbose=0)[0, 0]

        # Backpropagate value
        for node, move in reversed(path):
            child = node["children"][move]
            child["N"] += 1
            child["W"] += value
            child["Q"] = child["W"] / child["N"]
            node["N"] += 1

    def search(self, board):
        """
        Perform MCTS search and return the best move.
        Args:
            board: Current board state.
        Returns:
            Best move to take.
        """
        for _ in range(self.simulations):
            self.simulate(board)

        # Choose the move with the highest visit count
        root = self.tree[tuple(board.flatten())]
        best_move = max(root["children"], key=lambda move: root["children"][move]["N"])
        return best_move

    @staticmethod
    def get_valid_moves(board):
        """
        Get all valid moves for the current board state.
        Args:
            board: Current board state.
        Returns:
            List of valid moves.
        """
        moves = []
        for i in range(10):
            for j in range(10):
                if board[i, j] == 1:  # Blue piece
                    # Generate moves for blue pieces (up, down, left, right)
                    if i > 0: moves.append((i, j, i - 1, j))  # Up
                    if i < 9: moves.append((i, j, i + 1, j))  # Down
                    if j > 0: moves.append((i, j, i, j - 1))  # Left
                    if j < 9: moves.append((i, j, i, j + 1))  # Right
        return moves

    @staticmethod
    def apply_move(board, move):
        """
        Apply a move to the board.
        Args:
            board: Current board state.
            move: Move to apply (start_x, start_y, end_x, end_y).
        Returns:
            New board state.
        """
        start_x, start_y, end_x, end_y = move
        new_board = board.copy()
        new_board[end_x, end_y] = new_board[start_x, start_y]  # Move blue piece
        new_board[start_x, start_y] = 0  # Empty the old position
        return new_board

    @staticmethod
    def is_terminal(board):
        """
        Check if the game is in a terminal state.
        Args:
            board: Current board state.
        Returns:
            True if the game is over, otherwise False.
        """
        return np.count_nonzero(board == -1) == 0  # All red pieces are eliminated

    @staticmethod
    def evaluate_terminal(board):
        """
        Evaluate the value of a terminal state.
        Args:
            board: Current board state.
        Returns:
            +1 if blue wins, -1 otherwise.
        """
        if np.count_nonzero(board == -1) == 0:
            return 1  # All red pieces eliminated
        return -1  # Default loss condition
