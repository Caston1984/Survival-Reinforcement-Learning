# https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial
# https://github.com/geraudnt/boolean_composition/blob/master/four_rooms/exp3.py
from Gridworld import GridWorld
from SORL_Lib import *
import random
import os
from timeit import default_timer as timer
import gc

class Q_learning():
    def __init__(self, seed, goal_size,  directory, n_training_episodes=5, learning_rate=0.7, n_eval_episodes=100, gamma=0.95):
        # random seed
        self.seed = seed
        self.goal_size = goal_size
        random.seed(self.seed)

        # Training parameters
        self.n_training_episodes=n_training_episodes
        self.learning_rate=learning_rate

        # Evaluation parameters
        self.n_eval_episodes=n_eval_episodes

        # Environment parameters
        self.all_goal_pos = random.sample(self.goal_init(), goal_size)
        #print('All goals', self.all_goal_pos)
        self.loaded_goals = self.all_goal_pos[:10]
        #print('Loaded goals', self.loaded_goals)
        self.goals = [[pos, pos] for pos in self.all_goal_pos]
        self.start_pos = random.choice(self.goal_init())
        #print('Start position', self.start_pos)
        self.start_state = [random.randint(20, 35), random.randint(20, 35), self.start_pos[0], self.start_pos[1]]
        #print('Start state', self.start_state)
        self.goal_resources = self.resource_levels()
        #print('Rewards', self.goal_resources)
        self.gamma = gamma
        self.eval_seed = []

        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay_rate = 0.0005
        self.epsilon = 0

        self.resources = None
        self.start_position = None
        self.state = None
        self.step = None

        # Seeds
        self.seeds = [23, 51, 61, 94] # 5,

        # data directory
        self.directory = directory

        # States list and Q_table initialisation
        self.state_ls = self.states()
        self.state_space = len(self.state_ls)
        self.action_space = 5
        self.Q_table = self.initialize_q_table()

        # Output lists
        self.episode_rewards_t = []
        self.episode_length_t = []
        self.resource_t = []
        self.action_ls = []
        self.episode_time = []
        self.total_rewards_ep = 0

        self.frame = []

    def goal_init(self):
        env = GridWorld()
        goals = env.possiblePositions
        return goals

    def resource_levels(self):
        random.seed(self.seed)
        ls = []
        for i, goal in enumerate(self.all_goal_pos):
            if goal in self.loaded_goals:
                f = random.randrange(0, 4)
                w = random.randrange(0, 4)
                #if goal == self.loaded_goals[0]:
                 #   f = 5
                 #   w = 4
                #elif goal == self.loaded_goals[1]:
                 #   f = 36
                 #   w = 3
                #elif goal == self.loaded_goals[2]:
                 #   f = 2
                 #   w = 38
                #elif goal == self.loaded_goals[3]:
                 #   f = 2
                 #   w = 6
            else:
                f = 0
                w = 0
            res = [f, w]
            ls.append(res)
        return ls

    def states(self,):
        ls = []
        ls1 = []
        ls2 = []

        F = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
             23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
             77, 78, 79, 80, 81, 82]
        W = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
             23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
             50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
             77, 78, 79, 80, 81, 82]
        X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        for x in X:
            for y in Y:
                a = [x, y]
                ls.append(a)

        for w in W:
            for element in ls:
                ls1.append([w] + element)

        for f in F:
            for element in ls1:
                ls2.append([f] + element)

        return ls2

    # Initialising the Q-table
    def initialize_q_table(self):
        Qtable = np.zeros((self.state_space, self.action_space))
        return Qtable

    # Qtable_composition = initialize_q_table(state_space, action_space) # don't need it because I have already created it up there

    # Epsilon - greedy policy
    # problem is here when we pass the state to select an action, need to fix these issues
    def epsilon_greedy_policy(self, state_id):
        random_int = random.uniform(0, 1)
        if random_int > self.epsilon:
            action = np.argmax(self.Q_table[state_id])
        else:
            action = random.randrange(self.action_space)
        return action

    # greedy policy
    def greedy_policy(self, state_id):
        action = np.argmax(self.Q_table[state_id])
        return action

    def alive(self, ar):
        if ((ar[0] > 0) and (ar[0] < 40)) and ((ar[1] > 0) and (ar[1] < 42)):
            return True
        else:
            return False

    # Model Training
    def train(self,):
        for seed in self.seeds:
            # setting the random sead
            random.seed(seed)
            #print('Q-learning - seed', seed)
            self.episode_rewards_t = []
            self.episode_length_t = []
            self.resource_t = []
            self.action_ls = []
            self.Q_table = self.initialize_q_table()
            for episode in range(self.n_training_episodes):
                start = timer()
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                    -self.decay_rate * episode)
                total_rewards_ep = 0
                self.state = self.start_state
                self.start_position = (self.state[2], self.state[3])
                env = GridWorld(goals=self.goals, start_position=self.start_position)

                self.resources = np.array([self.state[0], self.state[1]])
                self.resource_t.append(self.resources)
                state_id = self.state_ls.index(self.state)
                self.step = 0

                while self.alive(self.resources):
                    action = self.epsilon_greedy_policy(state_id)
                    self.action_ls.append(action)
                    state_, step_reward, done, _ = env.step(action)
                    #env.render()

                    self.resources = np.subtract(self.resources, np.array([1, 1]))
                    self.resource_t.append(self.resources)
                    self.step += 1

                    if done:
                        idx = self.all_goal_pos.index(env.position)
                        self.resources = np.add(self.resources, np.array(self.goal_resources[idx]))

                        env = GridWorld(goals=self.goals, start_position=env.position)
                        self.resource_t.append(self.resources)

                    alive = self.alive(self.resources)

                    if not alive:
                        step_reward = step_reward - 1

                    new_position = env.position
                    new_state = [self.resources[0], self.resources[1], new_position[0], new_position[1]]

                    new_state_id = self.state_ls.index(new_state)

                    self.Q_table[state_id][action] = self.Q_table[state_id][action] + self.learning_rate * (
                            step_reward + self.gamma * np.max(self.Q_table[new_state_id]) - self.Q_table[state_id][action])

                    # Our state is the new state
                    state_id = new_state_id
                    # state = new_state
                    total_rewards_ep += step_reward

                #if (episode % 1000) == 0:
                 #   print('Q-learning Episode', episode, 'Total Reward', total_rewards_ep)
                self.episode_rewards_t.append(total_rewards_ep)
                self.episode_length_t.append(self.step)
                end = timer()
                self.episode_time.append(end - start)

            #print('saving for seed', seed)
            if not os.path.isdir(self.directory):
                os.mkdir(self.directory)
            file_1 = "ep_rewards_q_learning_%s_%r.npy" % (seed, self.goal_size)
            file_path = os.path.join(self.directory, file_1)
            np.save(file_path, self.episode_rewards_t)
            # file_0 = "ep_time_q_learning_%s_%r.npy" % (seed, self.goal_size)
            # file_path = os.path.join(self.directory, file_0)
            # np.save(file_path, self.episode_time)
            #file_2 = "ep_len_q_learning_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_2)
            #np.save(file_path, self.episode_length_t)
            #file_3 = "ep_res_q_learning_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_3)
            #np.save(file_path, self.resource_t)
            #file_4 = "ep_action_q_learning_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_4)
            #np.save(file_path, self.action_ls)

            # the Qtable array is saved in the file qlearning_qtable_seed.npy
            #file_5 = "q_learning_qtable_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_5)
            #np.save(file_path, self.Q_table)

        self.episode_time = None
        self.episode_rewards_t = None
        self.episode_length_t = None
        self.resource_t = None
        self.action_ls = None
        self.Q_table = None
        gc.collect()  # Force garbage collection

    