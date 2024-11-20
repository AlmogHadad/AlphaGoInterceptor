import numpy as np
from red_object import RedObject
from blue_object import BlueObject


class WorldManager:
    def __init__(self, red_object_list: list[RedObject], blue_object_list: list[BlueObject]):
        self.red_object_list: list[RedObject] = red_object_list
        self.blue_object_list: list[BlueObject] = blue_object_list

    def step(self, action):
        # red object step
        for red_object in self.red_object_list:
            red_object.step()

        # blue object step
        for idx, blue_object in enumerate(self.blue_object_list):
            blue_object.step(action[idx])

        return np.concatenate((self.blue_object_list[0].position, self.red_object_list[0].position)), 0, 0, {}, {}


# Define the state representation
class State:
    def __init__(self, blue_positions, blue_velocities, red_positions, red_velocities):
        self.blue_positions = blue_positions
        self.blue_velocities = blue_velocities
        self.red_positions = red_positions
        self.red_velocities = red_velocities

    def get_state_vector(self):
        return np.concatenate([
            self.blue_positions.flatten(),
            self.blue_velocities.flatten(),
            self.red_positions.flatten(),
            self.red_velocities.flatten()
        ])

    def next_state(self, action):
        # Placeholder: Update blue positions based on action
        action_effect = np.array([[action % 2, action // 2]])  # Example: Simple effect
        new_blue_positions = self.blue_positions + action_effect
        return State(new_blue_positions, self.blue_velocities, self.red_positions, self.red_velocities)


ACTIONS = {
    0: np.array([0, 1]),   # Move up
    1: np.array([0, -1]),  # Move down
    2: np.array([1, 0]),   # Move right
    3: np.array([-1, 0]),  # Move left
    4: np.array([0, 0])    # Stay
}


# Assuming best_action is a single action index
def apply_action(state, action_index):
    action = ACTIONS[action_index]  # Index the ACTIONS dictionary
    new_blue_positions = state.blue_positions + action
    # update the velocity of the blue object
    state.blue_velocities = action

    return State(new_blue_positions, state.blue_velocities, state.red_positions, state.red_velocities)