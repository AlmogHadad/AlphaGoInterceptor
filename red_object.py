

class RedObject:
    def __init__(self, position, velocity, trajectory=None):
        self.position = position
        self.velocity = velocity
        self.trajectory = trajectory
        self.current_trajectory_index = 0

    def step(self):
        if self.trajectory is not None:
            self.position = self.trajectory[self.current_trajectory_index]
            self.current_trajectory_index += 1
        else:
            self.position += self.velocity