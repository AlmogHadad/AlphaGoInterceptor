import numpy as np
import tensorflow as tf
from tensorflow import keras


# Train networks
def train_networks(policy_network, value_network, training_data, policy_lr=0.001, value_lr=0.001):
    policy_optimizer = keras.optimizers.Adam(policy_lr)
    value_optimizer = keras.optimizers.Adam(value_lr)

    states, policy_targets, value_targets = zip(*training_data)

    states = np.array(states)
    policy_targets = np.array(policy_targets)
    value_targets = np.array(value_targets)

    # Train policy network
    with tf.GradientTape() as tape:
        policy_predictions = policy_network(states)
        policy_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(policy_targets, policy_predictions))
    policy_grads = tape.gradient(policy_loss, policy_network.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, policy_network.trainable_variables))

    # Train value network
    with tf.GradientTape() as tape:
        value_predictions = value_network(states)
        # Use the MeanSquaredError instance properly
        mse_loss = keras.losses.MeanSquaredError()
        value_loss = mse_loss(value_targets, value_predictions)
    value_grads = tape.gradient(value_loss, value_network.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, value_network.trainable_variables))

    return policy_loss.numpy(), value_loss.numpy()


def create_policy_network(input_dim, output_dim):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(output_dim, activation='softmax')  # Softmax for probabilities
    ])
    return model


def create_value_network(input_dim):
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='linear')  # Single scalar output
    ])
    return model
