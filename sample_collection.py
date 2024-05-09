import numpy as np
import random

# Robot is named Just Samuel

class UnderwaterRobotEnv:
    def __init__(self, size=10, num_samples=4, num_tide_zones=4, initial_battery=100):
        self.size = size
        self.num_samples = num_samples
        self.num_tide_zones = num_tide_zones
        self.grid = np.zeros((size, size), dtype=int)
        self.robot_start = (0, 0)
        self.test_site = (size - 1, size - 1)
        self.robot_position = self.robot_start
        self.battery = initial_battery
        self.samples_collected = [False] * num_samples
        self.initialize_grid()

    def initialize_grid(self):
        # Set positions for robot start and testing site
        self.grid[self.robot_start] = 0
        self.grid[self.test_site] = 0
        
        all_positions = set((i, j) for i in range(self.size) for j in range(self.size))
        all_positions.discard(self.robot_start)
        all_positions.discard(self.test_site)

        # Random positions for samples
        self.sample_positions = random.sample(all_positions, self.num_samples)
        for pos in self.sample_positions:
            self.grid[pos] = 2
            all_positions.remove(pos)

        # Random positions for high tide zones (2x2)
        self.tide_positions = []
        for _ in range(self.num_tide_zones):
            while True:
                base = random.choice(list(all_positions))
                x, y = base
                if x < self.size-1 and y < self.size-1:
                    tide_zone = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
                    if all(pos in all_positions for pos in tide_zone):
                        self.tide_positions.extend(tide_zone)
                        for pos in tide_zone:
                            self.grid[pos] = 1
                            all_positions.remove(pos)
                        break

    def display_grid(self):
        print("Grid Layout:")
        print(self.grid)
        print("Robot Start:", self.robot_start)
        print("Test Site:", self.test_site)
        print("Sample Positions:", self.sample_positions)
        print("High Tide Positions:", self.tide_positions)

    def move_robot(self, direction):
        """ Move robot, considering possible failure and boundary issues """
        intended_position = self.calculate_new_position(direction)
        if random.random() < 0.9:
            new_position = intended_position
        else:
            alternate_direction = self.get_alternate_direction(direction)
            new_position = self.calculate_new_position(alternate_direction)
        
        if new_position:
            self.robot_position = new_position
        self.consume_battery()

    def calculate_new_position(self, direction):
        """ Calculate new position based on direction, adjusting for grid boundaries """
        moves = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
        delta_x, delta_y = moves[direction]
        new_x = self.robot_position[0] + delta_x
        new_y = self.robot_position[1] + delta_y

        # Check boundaries and adjust if out of grid
        if not (0 <= new_x < self.size and 0 <= new_y < self.size):
            # Move in the opposite direction instead
            opposite_direction = self.get_opposite_direction(direction)
            delta_x, delta_y = moves[opposite_direction]
            new_x = self.robot_position[0] + delta_x
            new_y = self.robot_position[1] + delta_y

        # Ensure the new position is still within bounds
        new_x = max(0, min(new_x, self.size - 1))
        new_y = max(0, min(new_y, self.size - 1))

        return (new_x, new_y)

    def get_alternate_direction(self, direction):
        """ Returns an alternate direction +/- 90 degrees to the attempted move """
        alternates = {
            'up': random.choice(['left', 'right']),
            'down': random.choice(['left', 'right']),
            'left': random.choice(['up', 'down']),
            'right': random.choice(['up', 'down'])
        }
        return alternates[direction]
    
    def get_opposite_direction(self, direction):
        """ Return opposite direction for boundary handling """
        opposites = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
        return opposites[direction]

    def consume_battery(self):
        """ Consume battery based on terrain """
        if self.grid[self.robot_position] == 1:
            self.battery -= 5
        else:
            self.battery -= 1

    def collect_sample(self):
        """ Collect a sample if it's at the robot's current position, consuming battery """
        self.consume_battery()
        if self.robot_position in self.sample_positions:
            index = self.sample_positions.index(self.robot_position)
            if not self.samples_collected[index]:
                self.samples_collected[index] = True
                return True
        return False

    def deposit_samples(self):
        """ Deposit all collected samples if at the test site, consuming battery """
        self.consume_battery()
        if self.robot_position == self.test_site:
            collected = any(self.samples_collected)
            self.samples_collected = [False] * len(self.samples_collected)
            return collected
        return False

# Initialize the environment
env = UnderwaterRobotEnv()

# Display initial settings
print("Initial Grid Layout:")
env.display_grid()

# Test movement and sample collection
print("\nTesting movement and sample collection:")
direction = ['right', 'right', 'down', 'down']
position = [(0,1), (0,2), (1,2), (2,2)]
for i in range(4):
    env.move_robot(direction[i])
    if env.robot_position == position[i]:
        print(f"Moved {direction[i]}. New Position: {env.robot_position}, Battery: {env.battery}")
    else:
        print(f"Failed to move {direction[i]}. Current Position: {env.robot_position}, Battery: {env.battery}")
    if env.collect_sample():
        print("Sample collected at:", env.robot_position)
    else:
        print("No sample to collect at:", env.robot_position)

# Test depositing samples
print("\nMoving to test site and trying to deposit samples:")
while env.robot_position != env.test_site:
    if env.robot_position[0] < env.test_site[0]:
        env.move_robot('down')
    elif env.robot_position[1] < env.test_site[1]:
        env.move_robot('right')
    print(f"Moving towards test site. Current Position: {env.robot_position}, Battery: {env.battery}")

if env.deposit_samples():
    print("Samples deposited at the test site.")
else:
    print("No samples to deposit or not at the test site.")

# Final state
print("\nFinal State:")
env.display_grid()
