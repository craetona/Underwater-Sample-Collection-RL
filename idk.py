
import numpy as np
import sys
import timeit


# Import time module
import time

FLAG = True


class Taxi_Grid():
    def __init__(self):
        self.total = {}
        self.ploted = {}
        self.size = 10
        self.grid_width = 10
        self.grid_height = 10
        self.actions = ['left','up','right','down','collect','deposit']
        self.traffic_state_id = [12,5,6,29,50,61,89,78,45,36,23,65,8,54,98,23,43,55,90,32,17,18,87,62,44,45]
        self.sample_id = [7,19,26,69,90]
        self.battery_level = 100
        self.gamma = 0.9
        self.test = set()
        self.policy = {}
        self.states = []
        self.has_sample = False
        self.collected_samples = set()
        self.count = 0

        self.populateParam()


    # Reward and transition function
    def populateParam(self):
        for s in range(100):
            self.states.append(s)
        self.s0 = 0
        self.goal_id = 99
        self.R =  np.full((len(self.states)), -1)

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

    def getSucc(self,s,action):
        succ_prob = 0.9
        fail_prob = 0.1
        succ = []
        if action == "left":
            if s%self.grid_width > 0:
                succ.append((s-1, succ_prob))
                if s > self.grid_width - 1: #slides up when its action fails
                    succ.append((s-self.grid_width,fail_prob))
                elif s <= self.grid_width and s >= 0: #slides down if it is the first row
                    succ.append((s+self.grid_width, fail_prob))
                return succ
            else:
                return None
        elif action == "up":
            if s >= self.grid_width:
                succ.append((s-self.grid_width,succ_prob))

                if s%(self.grid_width-1) > 0: #not right-most cell, slides rights when action fails
                    succ.append((s+1, fail_prob))
                elif s%(self.grid_width-1) == 0 and s > 0: #slides left if it is the last column
                    succ.append((s-1, fail_prob))
                return succ
            else:
                return None
        elif action == "right":
            if s%self.grid_width < 9:
                succ.append((s+1, succ_prob))
                if (s + self.grid_width) in self.states: #not the last row, slides down when action fails
                    succ.append(( s+ self.grid_width, fail_prob))
                elif s >= self.grid_width:  #moves up instead
                    succ.append(( s - self.grid_width, fail_prob))
                return succ 
            else:
                return None
        elif action == "down":
            if (s + self.grid_width) in self.states: #not the last row
                succ.append(( s+ self.grid_width, succ_prob))
                if s%self.grid_width > 0: #not the first column. Slides left when action fails
                    succ.append((s-1, fail_prob))
                elif s%(self.grid_width-1) >= 0:   #slides right instead
                    succ.append((s+1, fail_prob))
                return succ
            else:
                None

        elif action == "collect":
            if s in self.sample_id and s not in self.collected_samples:
                succ.append((s, 1))
                self.has_sample = True
                self.collected_samples.add(s)
                return succ
            else:
                return None


        elif action == "deposit":
            if s == self.goal_id and self.has_sample:
                succ.append((s, 1))
                return succ
            else:
                None

        return None
    
    def decrease_battery(self, s):
        if s in self.traffic_state_id:
            self.battery_level -= 5
        self.battery_level -= 1
        if self.battery_level < 0:
            self.battery_level = 0
    


    # FILL IN THE MISSING LINES TO SOLVE THE PROBLEM USING VI. 
    # Specifically, you are required to do the following:
    # 1. Calculate and print the time taken to solve VI
    # 2. Calculate Q value of an action
    # 3. Print policy
    # 4. Print value function

    def VI(self):
        self.V = np.zeros((len(self.states))).astype('float32').reshape(-1,1)
        self.Q =  np.zeros((len(self.states), len(self.actions))).astype('float32')
        max_trials = 10000
        epsilon = 0.000001
        initialize_bestQ = -10000
        curr_iter = 0
        bestAction = np.full((len(self.states)), -1)
        start = time.time()
        self.battery_level = 100
        while curr_iter < max_trials:
            print("CURR ITER: ", curr_iter, "---------------------------------"  )
            max_residual = 0
            curr_iter += 1
            self.collected_samples.clear()
            # Loop over states to calculate values
            # print(f"----------Iter {curr_iter}----------")
            
            for s in self.states:   
                if self.battery_level == 0:
                    self.V[s] -= 100
                    continue
                # print("Value of state", s, ":", self.V[s])
                if s == self.goal_id and self.has_sample == True:
                    bestAction[s] = 5
                    self.V[s] = 100 + 10*len(self.collected_samples)
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

                if FLAG == False:
                    continue

                

            if max_residual < epsilon:
                break

        self.policy = bestAction
        end = time.time()
        
#############################Complete the following line of code to print the time taken (in seconds) to solve VI
        print('Time taken to solve (seconds): ', (end-start) * 10**3, "ms")



    # Q-value calculation
    def qvalue(self,s, a):
        initialize_bestQ = -10000
        qaction = 0 #variable denoting the Q-value of the given (s,a) pair
        succ_list = self.P[s][a] 
        if succ_list is not None:
            reward = 0
            # if self.battery_level == 0:
            #     reward = -100
            if self.actions[a] == "collect":
                self.collected_samples.add(s)
                self.has_sample = True
                print("Collected:", self.collected_samples)
                reward = 10
            else:
                reward = self.R[s]
            qaction += reward
            for succ in succ_list:
                succ_state_id = self.states.index(succ[0]) #denotes s'
                prob = succ[1] #denotes the transition probability

#############################Complete the following line of code to calculate Q-value.
                qaction += prob * self.gamma * self.V[succ_state_id] #Q-Value formula
            return qaction
            

        else:
            return initialize_bestQ 

#############################Complete the following function to print the policy
    def printPolicy(self):
        for state in self.states:
            print(f"At state {state}: {self.actions[self.policy[state]]}")
#############################Complete the following function to print the value function
    def printValues(self):
        for s in self.states:
            print(f"State {s}: Value {self.V[s]}")

    def visualize_policy(self):
        """
        Visualize the policy in a 2D grid.

        Args:
            policy (numpy array): Policy array with shape (num_states, num_actions)
            grid_width (int): Width of the grid
            grid_height (int): Height of the grid
        """
        action_symbols = {
            0: '←',  # left
            1: '↑',  # up
            2: '→',  # right
            3: '↓',  # down
            4: 'C',  # collect
            5: 'D',  # deposit
            6: 'B'   # Out of battery
        }

        grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]

        for s in range(len(self.policy)):
            x, y = s % self.grid_width, s // self.grid_width
            action = self.policy[s]


            grid[y][x] = str(s) + action_symbols[action]

        for row in grid:
            print(' '.join(row))




taxi = Taxi_Grid()
taxi.VI()
taxi.printPolicy()
taxi.printValues()
taxi.visualize_policy()
