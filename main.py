import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from models import create_policy_network, create_value_network, train_networks
from world_manager import State, apply_action
from mcts import MCTS, MCTSNode


# Example workflow
if __name__ == "__main__":
    # Example setup
    blue_positions = np.array([[0, 0], [5, 5]])
    blue_velocities = np.array([[1, 0], [0, 1]])
    red_positions = np.array([[10, 10], [15, 15]])
    red_velocities = np.array([[0, -1], [-1, 0]])

    # Initialize state
    state = State(blue_positions, blue_velocities, red_positions, red_velocities)
    state_vector = state.get_state_vector()

    # Create networks
    input_dim = len(state_vector)
    action_dim = 5  # Example: 5 possible actions per object
    policy_network = create_policy_network(input_dim, action_dim)
    value_network = create_value_network(input_dim)

    # Initialize MCTS and collect training data
    training_data = []
    for episode in range(100):  # Simulate 100 episodes
        root = MCTSNode(state)
        mcts = MCTS(policy_network, value_network, action_dim)
        actions, visits = mcts.search(root, num_simulations=50)

        # Record state, policy target, and value target
        state_vector = root.state.get_state_vector()
        policy_target = np.array(visits) / sum(visits)
        value_target = np.max(policy_target)  # Example: use the max visit probability as value
        training_data.append((state_vector, policy_target, value_target))

    # Train the networks
    policy_loss, value_loss = train_networks(policy_network, value_network, training_data)
    print("Policy loss:", policy_loss)
    print("Value loss:", value_loss)

    # Initialize MCTS
    mcts = MCTS(policy_network, value_network, action_dim)

    # Root node
    root = MCTSNode(state)

    # Perform MCTS and apply actions in a loop
    for step in range(10):  # Run for a maximum of 10 steps
        actions, visits = mcts.search(root, num_simulations=50)
        print(f"Step {step}: Best action is {actions[np.argmax(visits)]}")

        # Apply the best action
        state = apply_action(state, actions[np.argmax(visits)])

        # Update root with the new state
        root = MCTSNode(state)

        # update the red objects position
        state.red_positions += state.red_velocities


        # Print the new state (for debugging)
        print("New blue positions:", state.blue_positions)
        print("Red positions:", state.red_positions)


        # Break if termination conditions are met (e.g., interception)
        if np.linalg.norm(state.blue_positions - state.red_positions, axis=1).min() < 1.0:
            print("Red objects intercepted!")
            break
