import numpy as np
import random
import sys

# Import time module
import time


# Robot is named Just Samuel

class UnderwaterRobotEnv:
    def __init__(self, size=10, num_samples=4, num_tide_zones=4, initial_battery=100):
        """
        I added the states array, actions, and tide_zones_id which is random
        """
        self.size = size
        self.states = []
        self.actions = ['left','up','right','down','collect','deposit']
        self.num_samples = num_samples
        self.num_tide_zones = num_tide_zones
        self.grid = np.zeros((size, size), dtype=int)
        self.tide_zones_id = [22,23,32,33,18,29,76,86,77,92,56,43,45,69,70,60]
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
        """
            I Think we need a predefined tide zone rather than random ones
        """


        # self.tide_positions = []
        # for _ in range(self.num_tide_zones):
        #     while True:
        #         base = random.choice(list(all_positions))
        #         x, y = base
        #         if x < self.size-1 and y < self.size-1:
        #             tide_zone = [(x, y), (x+1, y), (x, y+1), (x+1, y+1)]
        #             if all(pos in all_positions for pos in tide_zone):
        #                 self.tide_positions.extend(tide_zone)
        #                 for pos in tide_zone:
        #                     self.grid[pos] = 1
        #                     all_positions.remove(pos)
        #                 break


    # Reward and transition function
    def populateParam(self):
        for s in range(100):
            self.states.append(s)
        self.s0 = 0
        self.goal_id = 99
        self.R =  np.full((len(self.states)), -1)
        self.R[self.goal_id] = 100.0



        """
        Will need a transition function right here
        """
        self.P = [ [None]*len(self.actions) for i in range(len(self.states)) ]
        for s in self.states:
            for a, action in enumerate(self.actions):
                succ_sum=0
                self.P[s][a] = self.getSucc(s, action)
                if self.P[s][a]!=None:
                    for succ in self.P[s][a]:
                        succ_sum += succ[1]
                    if succ_sum !=1:
                        print(s,a,succ)
                        sys.exit()
    
    """
    Get Successor
    """
    def getSucc(self,s,action):
        succ_prob = 0.9
        fail_prob = 0.1
        succ = []
        if action == "left":
            if s%self.size > 0:
                succ.append((s-1, succ_prob))
                if s > self.size - 1: #slides up when its action fails
                    succ.append((s-self.size,fail_prob))
                elif s <= self.size and s >= 0: #slides down if it is the first row
                    succ.append((s+self.size, fail_prob))
                return succ
            else:
                return None
        elif action == "up":
            if s >= self.size:
                succ.append((s-self.size,succ_prob))

                if s%(self.size-1) > 0: #not right-most cell, slides rights when action fails
                    succ.append((s+1, fail_prob))
                elif s%(self.size-1) == 0 and s > 0: #slides left if it is the last column
                    succ.append((s-1, fail_prob))
                return succ
            else:
                return None
        elif action == "right":
            if s%self.size < 3:
                succ.append((s+1, succ_prob))
                if (s + self.size) in self.states: #not the last row, slides down when action fails
                    succ.append(( s+ self.size, fail_prob))
                elif s >= self.size:  #moves up instead
                    succ.append(( s - self.size, fail_prob))
                return succ 
            else:
                return None
        elif action == "down":
            if (s + self.size) in self.states: #not the last row
                succ.append(( s+ self.size, succ_prob))
                if s%self.size > 0: #not the first column. Slides left when action fails
                    succ.append((s-1, fail_prob))
                elif s%(self.size-1) >= 0:   #slides right instead
                    succ.append((s+1, fail_prob))
                return succ
            else:
                None
        elif action == "collect":
            succ.append((s,1))
        elif action == "deposit":
            succ.append((s,1))
        return None
    
    ## Value iteration
    def VI(self):
        self.V = np.zeros((len(self.grid))).astype('float32').reshape(-1,1)
        self.Q =  np.zeros((len(self.grid), len(self.actions))).astype('float32')
        max_trials = 100000
        epsilon = 0.00001
        initialize_bestQ = -10000
        curr_iter = 0
        bestAction = np.full((len(self.states)), -1)
        start = time.time()
        while curr_iter < max_trials:
            max_residual = 0
            curr_iter += 1

                
            # Loop over states to calculate values
            # print(f"----------Iter {curr_iter}----------")
            
            for s in self.states:   
                # print("Value of state", s, ":", self.V[s])
                if s == self.goal_id:
                    bestAction[s] = 0
                    self.V[s] = self.R[s]
                    continue
                bestQ = initialize_bestQ
                
                for na, a in enumerate(self.actions):
                    if self.P[s][na] is  None:
                        continue

                    qaction = max(initialize_bestQ, self.qvalue(s, na)) ##### Complete the code in "def qvalue() to calculate self.qvalue()
                    self.Q[s][na] = qaction 


                    if qaction > bestQ:
                        bestQ = qaction
                        bestAction[s] = na

                residual = abs(bestQ - self.V[s])
                self.V[s] = bestQ
                max_residual = max(max_residual, residual)

                

            if max_residual < epsilon:
                break

        self.policy = bestAction
        end = time.time()
        

        print('Time taken to solve (seconds): ', (end-start) * 10**3, "ms")


    # Q-value calculation
    def qvalue(self,s, a):
        initialize_bestQ = -10000
        qaction = 0 #variable denoting the Q-value of the given (s,a) pair
        succ_list = self.P[s][a] 
        if succ_list is not None:
            reward = self.R[s]
            qaction += reward
            for succ in succ_list:
                succ_state_id = self.states.index(succ[0]) #denotes s'
                prob = succ[1] #denotes the transition probability

                qaction += prob * self.gamma * self.V[succ_state_id] #Q-Value formula
            return qaction
            

        else:
            return initialize_bestQ 
    

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
