

class BlueObject:
    def __init__(self, position, current_velocity, speed):
        self.position = position
        self.current_velocity = current_velocity
        self.speed = speed

    def step(self, action):
        self.current_velocity = action
        self.position = self.position + self.current_velocity
