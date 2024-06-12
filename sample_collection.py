import numpy as np
import random
import timeit
import matplotlib.pyplot as plt
import pickle
import os

TERMINAL_FLAG = 0

class Underwater_Grid:
    def __init__(self, load_existing=True):
        print("Initializing Underwater_Grid")
        self.grid_width = 10
        self.grid_height = 10
        self.actions = ['left', 'right', 'up', 'down', 'collect', 'deposit']
        self.tide_region_id = [(0, 4), (4, 6), (5, 5), (2, 0), (2, 6),
                               (8, 3), (5, 3), (5, 0), (0, 4), (1, 4),
                               (1, 9), (2, 7), (8, 9), (7, 1), (9, 6),
                               (0, 6), (7, 6), (2, 5), (2, 2), (6, 0)]
        self.sample_id = {'one': (1, 6), 'two': (3, 0), 'three': (5, 7), 'four': (7, 2)}
        self.gamma = 0.9
        self.battery = 100

        if load_existing and os.path.exists('grid_data.pkl'):
            with open('grid_data.pkl', 'rb') as f:
                data = pickle.load(f)
                self.states = data['states']
                self.s0 = data['s']
                self.goal_id = data['goal_id']
                self.R = data['R']
                self.P = data['P']

        else:
            self.states = []
            self.R = {}
            self.populateParam()
            with open('grid_data.pkl', 'wb') as f:
                data = {'states': self.states, 's': self.s0, 'goal_id': self.goal_id, 'R': self.R, 'P': self.P}
                pickle.dump(data, f)

        self.Q = [[0] * len(self.actions) for i in range(len(self.states))]
        self.V = np.zeros(len(self.states))
        self.pi = [ [1 / len(self.states)]*len(self.actions) for _ in range(len(self.states)) ]
        print("Initialization complete")

    def populateParam(self):
        print("Populating parameters")
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                for b in range(self.battery + 1):
                    for s1 in [0, 1]:
                        for s2 in [0, 1]:
                            for s3 in [0, 1]:
                                for s4 in [0, 1]:
                                    self.states.append((x, y, b, (s1, s2, s3, s4)))
        print(f"State space created with {len(self.states)} states")
        
        self.s0 = (0, 0, self.battery, (0, 0, 0, 0))
        self.goal_id = []

        for b in range(self.battery + 1):
            for s1 in [0, 1]:
                for s2 in [0, 1]:
                    for s3 in [0, 1]:
                        for s4 in [0, 1]:
                            self.goal_id.append((9, 9, b, (s1, s2, s3, s4)))
        for b in range(self.battery + 1):
            self.goal_id.remove((9, 9, b, (0, 0, 0, 0)))
        print(f"Goal states created with {len(self.goal_id)} states")
        
        for state in self.states:
            for action in self.actions:
                if (state[0], state[1]) in self.tide_region_id:
                    self.R[(state, action)] = -10
                elif action == 'deposit' and state in self.goal_id:
                    self.R[(state, action)] = 25 * sum(state[3])
                elif action == 'collect' and (state[0], state[1]) in self.sample_id.values():
                    sample_index = list(self.sample_id.values()).index((state[0], state[1]))
                    if state[3][sample_index] == 0:
                        self.R[(state, action)] = 10
                    else:
                        self.R[(state, action)] = -1
                else:
                    self.R[(state, action)] = -1
        print(f"Reward function created with {len(self.R)} states")

        self.P = [ [None]*len(self.actions) for i in range(len(self.states)) ]

        for s, state in enumerate(self.states):
            if s % 10000 == 0:
                print(f"Generated successors, probabilities {s} out of {len(self.states)}")
            for a, action in enumerate(self.actions):
                idx = self.states.index(state)
                self.P[idx][a] = self.getSucc(state, action)

        print("Parameters populated")

    def get_reward(self, state, action):
        return self.R.get((state, action), -1)
    
    def consume_battery(self, x, y, b):
        if (x, y) in self.tide_region_id:
            return max(0, b - 5)
        else:
            return max(0, b - 1)
    
    def getSucc(self, state, action):
        succ_prob = 0.9
        fail_prob = 0.1
        succ = []

        if state[2] == 0:
            return None
        
        new_battery = self.consume_battery(state[0], state[1], state[2])

        if action == 'left':
            if state[0] % self.grid_width > 0:
                succ.append(((state[0] - 1, state[1], new_battery, state[3]), succ_prob))
                if state[1] % self.grid_height > 0:
                    succ.append(((state[0], state[1] - 1, new_battery, state[3]), fail_prob))
                else:
                    succ.append(((state[0], state[1] + 1, new_battery, state[3]), fail_prob))
                return succ
            else:
                return None
        
        elif action == 'right':
            if state[0] % self.grid_width < 9:
                succ.append(((state[0] + 1, state[1], new_battery, state[3]), succ_prob))
                if state[1] % self.grid_height < 9:
                    succ.append(((state[0], state[1] + 1, new_battery, state[3]), fail_prob))
                else:
                    succ.append(((state[0], state[1] - 1, new_battery, state[3]), fail_prob))
                return succ
            else:
                return None
            
        elif action == 'up':
            if state[1] % self.grid_height > 0:
                succ.append(((state[0], state[1] - 1, new_battery, state[3]), succ_prob))
                if state[0] % self.grid_width < 9:
                    succ.append(((state[0] + 1, state[1], new_battery, state[3]), fail_prob))
                else:
                    succ.append(((state[0] - 1, state[1], new_battery, state[3]), fail_prob))
                return succ
            else:
                return None
            
        elif action == 'down':
            if state[1] % self.grid_height < 9:
                succ.append(((state[0], state[1] + 1, new_battery, state[3]), succ_prob))
                if state[0] % self.grid_width > 0:
                    succ.append(((state[0] - 1, state[1], new_battery, state[3]), fail_prob))
                else:
                    succ.append(((state[0] + 1, state[1], new_battery, state[3]), fail_prob))
                return succ
            else:
                return None
            
        elif action == 'collect':
            if (state[0], state[1]) in self.sample_id.values():
                sample_idx = list(self.sample_id.values()).index((state[0], state[1]))
                if state[3][sample_idx] == 0:
                    new_collected_samples = list(state[3])
                    new_collected_samples[sample_idx] = 1
                    new_collected_samples = tuple(new_collected_samples)
                    succ.append(((state[0], state[1], new_battery, new_collected_samples), 1.0))
            else:
                succ.append(((state[0], state[1], new_battery, state[3]), 1.0))
            return succ
        
        elif action == 'deposit':
            if state in self.goal_id:
                TERMINAL_FLAG = 1
            succ.append(((state[0], state[1], new_battery, state[3]), 1.0))
            return succ
        

    def generateRandomSuccessor(self, state, action):
        random_value = random.uniform(0, 1)
        prob_sum = 0
        if state in self.goal_id:
            return state
        
        state_index = self.states.index(state)
        action_index = self.actions.index(action)

        if self.P[state_index][action_index] is None:
            print(f"No successors found for state {state} and action {action}")
            return state

        for succ in self.P[state_index][action_index]:
            succ_state = succ[0]
            prob = succ[1]
            prob_sum += prob
            if prob_sum >= random_value:
                return succ_state
            
        return state
            
    def get_epsilon_greedy_action(self, state, epsilon):
        applicable_actions = []
        pi_s_a = np.zeros((len(self.actions))).astype('float32').reshape(-1,1)
        best_Q = float('-inf')
        best_action = None
        idx = self.states.index(state)
        for a, action in enumerate(self.actions):
            if self.P[idx][a] is not None:
                applicable_actions.append(action)
                if self.Q[idx][a] > best_Q:
                    best_Q = self.Q[idx][a]
                    best_action = action

        for a in applicable_actions:
            if a == best_action:
                pi_s_a[self.actions.index(a)] = 1 - epsilon + (epsilon / len(applicable_actions))
            else:
                pi_s_a[self.actions.index(a)] = epsilon / len(applicable_actions)

        random_value = random.uniform(0, 1)

        prob = 0
        for action in applicable_actions:
            prob += pi_s_a[self.actions.index(action)]
            if prob >= random_value:
                return action
    
    # monte carlo with epsilon decay to try and maximize performance
    def every_visit_monte_carlo(self, num_episodes, epsilon, epsilon_decrement, decrement_frequency):
        returns_sum = {}
        returns_count = {}
        total_rewards = []
        runtimes = []

        for state in self.states:
            for action in self.actions:
                returns_sum[(state, action)] = 0.0
                returns_count[(state, action)] = 0

        for episode in range(num_episodes):
            if episode % decrement_frequency == 0:
                epsilon = max(0.01, epsilon - epsilon_decrement)

            print("***************************** Episode:", episode + 1)
            start_time = timeit.default_timer()
            episode_data = []
            state = self.s0
            total_reward = 0
            while True:
                action = self.get_epsilon_greedy_action(state, epsilon)
                if action is None:
                    break
                next_state = self.generateRandomSuccessor(state, action)
                reward = self.get_reward(state, action)
                episode_data.append((state, action, reward))
                total_reward += reward
                if next_state is None:
                    break
                if TERMINAL_FLAG:
                    print(f"Goal state reached: {state}")
                    break
                state = next_state

            total_rewards.append(total_reward)
            G = 0
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = reward + self.gamma * G
                if (state, action) not in [(x[0], x[1]) for x in episode_data[:t]]:
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1
                    self.Q[self.states.index(state)][self.actions.index(action)] = returns_sum[(state, action)] / returns_count[(state, action)]

            for state, action, _ in episode_data:
                best_action = np.argmax(self.Q[self.states.index(state)])
                for a in self.actions:
                    if a == self.actions[best_action]:
                        self.pi[self.states.index(state)][self.actions.index(a)] = 1 - epsilon + epsilon / len(self.actions)
                    else:
                        self.pi[self.states.index(state)][self.actions.index(a)] = epsilon / len(self.actions)
            runtimes.append(start_time - timeit.default_timer())
        
        return total_rewards, runtimes
    
    # monte carlo without epsilon decay for hyperparameter searching
    def monte_carlo(self, num_episodes, epsilon):
        returns_sum = {}
        returns_count = {}
        total_rewards = []
        runtimes = []

        for state in self.states:
            for action in self.actions:
                returns_sum[(state, action)] = 0.0
                returns_count[(state, action)] = 0

        for episode in range(num_episodes):
            print("***************************** Episode:", episode + 1)
            start_time = timeit.default_timer()
            episode_data = []
            state = self.s0
            total_reward = 0
            while True:
                action = self.get_epsilon_greedy_action(state, epsilon)
                if action is None:
                    break
                next_state = self.generateRandomSuccessor(state, action)
                reward = self.get_reward(state, action)
                episode_data.append((state, action, reward))
                total_reward += reward
                if next_state is None:
                    break
                if TERMINAL_FLAG:
                    print(f"Goal state reached: {state}")
                    break
                state = next_state

            total_rewards.append(total_reward)
            G = 0
            for t in range(len(episode_data) - 1, -1, -1):
                state, action, reward = episode_data[t]
                G = reward + self.gamma * G
                if (state, action) not in [(x[0], x[1]) for x in episode_data[:t]]:
                    returns_sum[(state, action)] += G
                    returns_count[(state, action)] += 1
                    self.Q[self.states.index(state)][self.actions.index(action)] = returns_sum[(state, action)] / returns_count[(state, action)]

            for state, action, _ in episode_data:
                best_action = np.argmax(self.Q[self.states.index(state)])
                for a in self.actions:
                    if a == self.actions[best_action]:
                        self.pi[self.states.index(state)][self.actions.index(a)] = 1 - epsilon + epsilon / len(self.actions)
                    else:
                        self.pi[self.states.index(state)][self.actions.index(a)] = epsilon / len(self.actions)
            runtimes.append(start_time - timeit.default_timer())
        
        return total_rewards, runtimes
    
    def value_iteration(self, theta=1e-6, max_iterations=1000):
        total_rewards = []
        
        for i in range(max_iterations):
            delta = 0
            total_reward = 0
            for s, state in enumerate(self.states):
                v = self.V[s]
                max_value = float('-inf')
                for a, action in enumerate(self.actions):
                    q_value = 0
                    successors = self.P[s][a]
                    if successors is not None:
                        for next_state, prob in successors:
                            reward = self.get_reward(state, action)
                            next_state_index = self.states.index(next_state)
                            q_value += prob * (reward + self.gamma * self.V[next_state_index])
                    if q_value > max_value:
                        max_value = q_value
                self.V[s] = max_value
                delta = max(delta, abs(v - self.V[s]))
                total_reward += self.V[s]
            
            total_rewards.append(total_reward)
            
            if delta < theta:
                break

        for s, state in enumerate(self.states):
            max_value = float('-inf')
            best_action = None
            for a, action in enumerate(self.actions):
                q_value = 0
                successors = self.P[s][a]
                if successors is not None:
                    for next_state, prob in successors:
                        reward = self.get_reward(state, action)
                        next_state_index = self.states.index(next_state)
                        q_value += prob * (reward + self.gamma * self.V[next_state_index])
                if q_value > max_value:
                    max_value = q_value
                    best_action = a
            self.Q[s] = [0] * len(self.actions)
            self.Q[s][best_action] = max_value
        
        return total_rewards

    def save_policy(self, filename):
        state = self.s0
        with open(filename, 'w') as f:
            while state not in self.goal_id:
                best_action = np.argmax(self.Q[self.states.index(state)])
                action_name = self.actions[best_action]
                f.write(f"State: {state}, Action: {action_name}\n")
                next_state = self.generateRandomSuccessor(state, action_name)
                if next_state is None:
                    break
                state = next_state
    

#hyper parameter search on monte carlo
all_rewards = []
best_rewards = []
best_runtimes = []
best_epsilon = 0
epsilons = np.linspace(0.01, 0.9, 20)

for epsilon in epsilons:
    print(f"Testing epsilon {epsilon}")
    grid = Underwater_Grid()
    rewards, runtimes = grid.monte_carlo(3000, epsilon)
    if rewards:
        avg_reward = np.mean(rewards)
        print(f"Epsilon {epsilon} average reward: {avg_reward}")
        all_rewards.append((epsilon, avg_reward))
        if not best_rewards or avg_reward > np.mean(best_rewards):
            best_rewards = rewards
            best_runtimes = runtimes
            best_epsilon = epsilon

# Best hyperparameters found
print("Optimized epsilon:", best_epsilon)
print("Average runtime per episode:", np.mean(best_runtimes))
for i in range(len(all_rewards)):
    print(f"Average reward per epsilon: {epsilons[i]}: {all_rewards[i]}")

# Using matplotlib to plot the total rewards per episode
plt.figure(figsize=(16, 8))
plt.plot(best_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Monte Carlo Total Reward per Episode")
plt.grid(True)
plt.show()


# Monte Carlo with epsilon decay
grid = Underwater_Grid()
rewards, runtimes = grid.every_visit_monte_carlo(10000, 0.8, 0.18, 1500)
avg_reward = np.mean(rewards)
print(f"Average reward: {avg_reward}")
best_rewards = rewards
best_runtimes = runtimes

# Using matplotlib to plot the total rewards per episode
plt.figure(figsize=(16, 8))
plt.plot(best_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Monte Carlo Total Reward per Episode")
plt.grid(True)
plt.show()


# Run value iteration and save the learned policy
grid = Underwater_Grid()
total_rewards = grid.value_iteration()
grid.save_policy("learned_policy.txt")

plt.figure(figsize=(16, 8))
plt.plot(total_rewards)
plt.xlabel("Iteration")
plt.ylabel("Total Reward")
plt.title("Value Iteration Total Reward Accumulated Over Time")
plt.grid(True)
plt.show()
