import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


# Train networks
def train_networks(policy_net, value_net, board_states, expert_moves, game_outcomes,
                   board_size=9, epochs=10, batch_size=32):
    """
    Train the policy and value networks.

    Args:
        policy_net (tf.keras.Model): Policy network to train.
        value_net (tf.keras.Model): Value network to train.
        board_states (np.ndarray): Training data of board states (shape: [n_samples, board_size, board_size, 1]).
        expert_moves (np.ndarray): Target moves for the policy network (shape: [n_samples]).
        game_outcomes (np.ndarray): Target outcomes for the value network (-1 for loss, 1 for win, 0 for draw).
        board_size (int): The size of the Go board.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.

    Returns:
        history (dict): Training histories for the policy and value networks.
    """
    # Prepare policy network labels (one-hot encoded moves)
    policy_labels = keras.utils.to_categorical(expert_moves, num_classes=board_size * board_size)

    # Compile policy network
    policy_net.compile(optimizer="adam",
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])

    # Compile value network
    value_net.compile(optimizer="adam",
                      loss="mse",  # Mean Squared Error for regression
                      metrics=["mae"])  # Mean Absolute Error

    # Train policy network
    print("Training policy network...")
    policy_history = policy_net.fit(
        board_states, policy_labels,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Train value network
    print("Training value network...")
    value_history = value_net.fit(
        board_states, game_outcomes,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Return training histories
    return {"policy_history": policy_history, "value_history": value_history}


def create_policy_network(board_size=10):
    inputs  = keras.layers.Input(shape=(board_size, board_size, 1))  # Board state as input
    x       = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x       = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x       = keras.layers.Flatten()(x)
    x       = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(board_size*board_size, activation='softmax')(x)  # Probability for each move
    return keras.Model(inputs, outputs)

def create_value_network(board_size=10):
    inputs  = keras.layers.Input(shape=(board_size, board_size, 1))  # Board state as input
    x       = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x       = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x       = keras.layers.Flatten()(x)
    x       = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation='tanh')(x)  # Value in range [-1, 1]
    return keras.Model(inputs, outputs)