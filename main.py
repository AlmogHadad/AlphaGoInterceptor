import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from models import create_policy_network, create_value_network, train_networks
from world_manager import State, apply_action
from mcts import MCTS


# Example workflow
if __name__ == "__main__":
    # Step 1: Initialize Neural Networks
    print("Initializing policy and value networks...")
    policy_network = create_policy_network()
    value_network = create_value_network()

    # Compile the models
    policy_network.compile(optimizer='adam', loss='categorical_crossentropy')
    value_network.compile(optimizer='adam', loss='mse')

    # Step 2: Create MCTS Instance
    print("Initializing Monte Carlo Tree Search...")
    mcts = MCTS(policy_network, value_network, simulations=100)

    # Step 3: Set Up the Game Environment
    print("Setting up the game environment...")
    board = np.zeros((10, 10))  # Initialize a 10x10 empty board
    # Add initial positions for blue and red objects
    board[0, 0] = 1  # Example: Place one blue object
    board[9, 9] = -1  # Example: Place one red object

    print("Initial board state:")
    print(board)

    # Step 4: Play a Game
    print("Starting the game...")
    turn = 0  # 0 for blue (player), 1 for red (AI or simulation)
    while not mcts.is_terminal(board):
        print(f"Turn {'Blue' if turn == 0 else 'Red'}:")
        if turn == 0:  # Blue player's turn
            move = mcts.search(board)  # Use MCTS to decide the best move
            print(f"Blue moves: {move}")
            board = mcts.apply_move(board, move)
        else:  # Red's turn (for now, red doesn't actively play)
            print("Red does nothing (example logic).")

        print("Current board state:")
        print(board)

        # Switch turn
        turn = 1 - turn

    # Step 5: Determine the Winner
    if np.count_nonzero(board == -1) == 0:
        print("Blue wins! All red objects have been eliminated.")
    else:
        print("Red wins or the game ended in a draw.")