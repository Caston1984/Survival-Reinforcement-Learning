# https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial
# https://github.com/geraudnt/boolean_composition/blob/master/four_rooms/exp3.py

from Gridworld import GridWorld
from Library import *
import random
import numpy as np
import os
from timeit import default_timer as timer
import gc


class Q_learn_composition():
    def __init__(self, seed, goal_size, base_goals_size, directory, n_training_episodes=5, learning_rate=0.7, n_eval_episodes=100, gamma=0.95):
        # random seed
        self.seed = seed
        self.goal_size = goal_size
        random.seed(self.seed)

        # Training parameters
        self.n_training_episodes = n_training_episodes
        self.learning_rate = learning_rate

        # Evaluation parameters
        self.n_eval_episodes = n_eval_episodes
        self.base_goal_size = base_goals_size

        # Environment parameters
        self.all_goal_pos = random.sample(self.goal_init(), goal_size) # Initialise the whole possible grid state's space as goals[(g1_x, g1_y), (g2_x, g2_y), (g3_x, g3_y), (g4_x, g4_y)]
        print('All goal', self.all_goal_pos)
        self.loaded_goals = self.all_goal_pos[:10]
        #print('Loaded goals', self.loaded_goals)
        self.start_pos = random.choice(self.goal_init())
        print('Start position', self.start_pos)
        self.start_state = [random.randint(20, 35), random.randint(20, 35), self.start_pos[0], self.start_pos[1]]
        print('Start state', self.start_state)
        self.base_1, self.base_2 = self.base_constructor()
        print(self.boolean_exp())
        #self.base_1 = [(1, 1), (1, 10)]
                        #[(1, 10), (10, 10)]  #random.sample(self.all_goal_pos, base_goals_size)
        #self.base_2 = [(1, 1), (10, 10)]
                        #[(2, 1), (10, 1)]  #random.sample(self.all_goal_pos, base_goals_size)
        print('Base Tasks', self.base_1, self.base_2)
        self.goal_resources = self.resource_levels()
        print('Rewards', self.goal_resources)
        self.gamma = gamma
        self.eval_seed = []

        # Exploration parameters
        self.max_epsilon = 1.0
        self.min_epsilon = 0.05
        self.decay_rate = 0.0005
        self.epsilon = 0

        self.T_states = None
        self.tasks = None
        self.start_position = None
        self.resources = None
        self.step = 0
        self.action = 0

        # Seeds
        self.seeds = [23, 51, 61, 94] #5,

        # data directory
        self.directory = directory

        # States list and Q_table initialisation
        self.state_ls = self.states()
        self.state_space = len(self.state_ls)
        self.action_space = 14
        self.Q_table = self.initialize_q_table()

        # Output lists
        self.episode_rewards_t = []
        self.episode_length_t = []
        self.resource_t = []
        self.action_ls = []
        self.episode_time = []
        self.total_rewards_ep = 0

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

    # a1=False, a2=False, a3=False, a4=False, a5=False, a6=False, a7=False, a8=False, a9=False,
    #                     a10=False, a11=False, a12=False, a13=False, a14=False
    def boolean_exp(self,):
        ls = []
        # AND(A, B)
        if not list(set(self.base_1) & set(self.base_2)):
            ls.append('AND(A, B)')

        # AND(A, NEG(B))
        if not list(set(self.base_1) - set(self.base_2)):
            ls.append('AND(A, NEG(B))')

        # AND(NEG(A), B)
        if not list(set(self.base_2) - set(self.base_1)):
            ls.append('AND(NEG(A), B) ')

        # NEG(OR(A, B))
        u = set(self.base_1) | set(self.base_2)
        if not list(set(self.all_goal_pos) - u):
            ls.append('NEG(OR(A, B))')

        # A
        if not list(set(self.base_1)):
            ls.append('A')

        # NEG(A)
        if not list(set(self.all_goal_pos) - set(self.base_1)):
            ls.append('NEG(A)')

        # B
        elif not list(set(self.base_2)):
            ls.append('B')

        # NEG(B)
        elif not list(set(self.all_goal_pos) - set(self.base_2)):
            ls.append('NEG(B)')

        # OR(A, B)
        if not list(set(self.base_1) | set(self.base_2)):
            ls.append('OR(A, B)')

        # OR(A, NEG(B))
        if not list(set(self.all_goal_pos) - set(self.base_2)):
            ls.append('OR(A, NEG(B))')

        # OR(B, NEG(A))
        if not list(set(self.all_goal_pos) - set(self.base_1)):
            ls.append('OR(B, NEG(A))')

        # NEG(AND(A, B))
        y = set(self.base_1) & set(self.base_2)
        if not list(set(self.all_goal_pos) - y):
            ls.append('# NEG(AND(A, B))')

        # NEG(XOR(A, B))
        z = set(self.base_1) ^ set(self.base_2)
        if not list(set(self.all_goal_pos) - z):
            ls.append('NEG(XOR(A, B))')

        # XOR(A, B)
        if not list(set(self.base_1) ^ set(self.base_2)):
            ls.append('XOR(A, B)')

        if not ls:
            return print('All compositions are there')
        else:
            return print('Missing the following compositions', ls)

    def a_and_b(self):
        # AND(A, B)
        return list(set(self.base_1) & set(self.base_2))

    def a_and_not_b(self):
        # AND(A, NEG(B))
        return list(set(self.base_1) - set(self.base_2))

    def b_and_not_a(self):
        # AND(NEG(A), B)
        return list(set(self.base_2) - set(self.base_1))

    def not_a_or_b(self):
        # NEG(OR(A, B))
        u = set(self.base_1) | set(self.base_2)
        return list(set(self.all_goal_pos) - u)

    def a(self):
        # A
        return list(self.base_1)

    def not_a(self):
        # NEG(A)
        return list(set(self.all_goal_pos) - set(self.base_1))

    def b(self):
        # B
        return list(self.base_2)

    def not_b(self):
        # NEG(B)
        return list(set(self.all_goal_pos) - set(self.base_2))

    def a_or_b(self):
        # OR(A, B)
        return list(set(self.base_1) | set(self.base_2))

    def a_or_not_b(self):
        # OR(A, NEG(B))
        return list(set(self.all_goal_pos) - set(self.base_2))

    def b_or_not_a(self):
        # OR(B, NEG(A))
        return list(set(self.all_goal_pos) - set(self.base_1))

    def not_a_and_b(self):
        # NEG(AND(A, B))
        y = set(self.base_1) & set(self.base_2)
        return list(set(self.all_goal_pos) - y)

    def not_a_xor_b(self):
        # NEG(XOR(A, B))
        z = set(self.base_1) ^ set(self.base_2)
        return list(set(self.all_goal_pos) - z)

    def a_xor_b(self):
        # XOR(A, B)
        return list(set(self.base_1) ^ set(self.base_2))

    def base_constructor(self):
        base_1 = random.sample(self.loaded_goals, k=7)
        base_2 = random.sample(self.loaded_goals, k=7)

        return base_1, base_2

    def states(self):
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

    # the composition function
    def composition(self):
        # Later to move it to 1000 once I have learnt to save to improve quality
        maxiter = 2000
        #print('all goals', self.all_goal_pos)

        T_states = self.all_goal_pos
        self.T_states = [[pos, pos] for pos in T_states]
        #print('Terminal States', self.T_states)

        #Bases = [self.base_1, self.base_2]
        self.tasks = [self.a_and_b(), self.a_and_not_b(), self.b_and_not_a(),
                      self.not_a_or_b(), self.a(), self.not_a(),
                      self.b(), self.not_b(), self.a_or_b(),
                      self.a_or_not_b(), self.b_or_not_a(), self.not_a_and_b(),
                      self.not_a_xor_b(), self.a_xor_b()]


        # Learning universal bounds (min and max tasks)
        #print('Learning the max task')
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=self.T_states, dense_rewards=False)
        EQ_max, stats, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
        self.episode_time.extend(time_ls1)

        #print('Learning the min task')
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=self.T_states, dense_rewards=False)
        EQ_min, stats, time_ls2 = Goal_Oriented_Q_learning(env, maxiter=maxiter)
        self.episode_time.extend(time_ls2)

        # Learning base tasks and doing composed tasks
        goals = [[pos, pos] for pos in self.base_1]
        #print('Base 1 goals', goals)
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=goals, dense_rewards=False,
                        T_states=self.T_states)
        A, stats1, time_ls3 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=self.T_states)
        self.episode_time.extend(time_ls3)

        #print('Learning the second base task', self.base_2)
        goals = [[pos, pos] for pos in self.base_2]
        #print('Base 2 goals', goals)
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=goals, dense_rewards=False,
                        T_states=self.T_states)
        B, stats2, time_ls4 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=self.T_states)
        self.episode_time.extend(time_ls4)

        return EQ_max, EQ_min, A, B

    def alive(self, ar):
        if ((ar[0] > 0) and (ar[0] < 40)) and ((ar[1] > 0) and (ar[1] < 42)):
            return True
        else:
            return False

    def evaluate(self, goals, EQ, slip_prob=0):
        env = GridWorld(goals=goals, start_position=self.start_position,
                        slip_prob=slip_prob)
        policy = EQ_P(EQ)
        state = env.reset()
        done = False
        t = 0
        reward = 0
        self.step = 0
        while not done and t < 10000 and self.alive(self.resources):
            action = policy[state]
            state, step_reward, done, _ = env.step(action)
            #env.render()
            reward = reward + step_reward
            self.resources = np.subtract(self.resources, np.array([1, 1]))
            self.resource_t.append(self.resources)
            self.step += 1
            t += 1

        if done:
            idx = self.all_goal_pos.index(env.position)
            self.resources = np.add(self.resources, np.array(self.goal_resources[idx]))

        alive = self.alive(self.resources)

        if not alive:
            reward = reward - 1

        return reward, env.position

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

    # Model Training
    def train(self):
        # Composition of the base functions
        EQ_max, EQ_min, A, B = self.composition()
        NEG = lambda x: NOT(x, EQ_max=EQ_max, EQ_min=EQ_min)
        XOR = lambda EQ1, EQ2: OR(AND(EQ1, NEG(EQ2)), AND(EQ2, NEG(EQ1)))
        composed = [AND(A, B), AND(A, NEG(B)), AND(B, NEG(A)), NEG(OR(A, B)), A, NEG(A), B, NEG(B),
                    OR(A, B), OR(A, NEG(B)), OR(B, NEG(A)), NEG(AND(A, B)), NEG(XOR(A, B)), XOR(A, B)]

        base_training_episodes = len(self.episode_time)
        for seed in self.seeds:
            # setting the random sead
            random.seed(seed)
            #print('Composition RL - seed', seed)
            self.episode_rewards_t = []
            self.episode_length_t = []
            self.resource_t = []
            self.action_ls = []
            self.Q_table = self.initialize_q_table()
            #self.episode_rewards_t = [-100] * base_training_episodes
            #- base_training_episodes
            for episode in range(self.n_training_episodes):
                start = timer()
                # epsilon = 1
                self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(
                    -self.decay_rate * episode)

                self.total_rewards_ep = 0

                self.state = self.start_state
                self.start_position = (self.state[2], self.state[3])
                self.resources = np.array([self.state[0], self.state[1]])
                self.step = 0

                while self.alive(self.resources):
                    state_id = self.state_ls.index(self.state)
                    self.action = self.epsilon_greedy_policy(state_id)
                    self.action_ls.append(self.action)
                    # new_state, reward, done, info = env.step(action)
                    # the selected action is the index for the composition to be executed
                    goals = [[pos, pos] for pos in self.tasks[self.action]]
                    #print('Action', self.action, 'goals selected', goals)
                    # Evaluate the composition policy selected
                    reward, position = self.evaluate(goals, composed[self.action])
                    self.total_rewards_ep += reward
                    #print('reward', reward, 'Total rewards', self.total_rewards_ep)

                    new_state = []
                    for i in self.resources:
                        # appending resources
                        new_state.append(i)
                    for i in position:
                        new_state.append(i)

                    new_state_id = self.state_ls.index(new_state)

                    self.Q_table[state_id][self.action] = (self.Q_table[state_id][self.action] + self.learning_rate *
                                                           (reward + self.gamma * np.max(self.Q_table[new_state_id]) -
                                                            self.Q_table[state_id][self.action]))

                    #print('Q-value row for', state_id, 'is as follows', self.Q_table[state_id])
                    # Assign the new state as the current
                    self.state = new_state

                    # If alive current position is start position
                    if self.alive(self.resources):
                        self.start_position = position

                #if (episode % 1000) == 0:
                 #   print('Composition RL Episode', episode, 'Total Reward', self.total_rewards_ep)
                #print('episode', episode, 'Total rewards', self.total_rewards_ep)
                self.episode_rewards_t.append(self.total_rewards_ep)
                self.episode_length_t.append(self.step)
                end = timer()
                self.episode_time.append(end - start)

            #print('saving for seed', seed)
            if not os.path.isdir(self.directory):
                os.mkdir(self.directory)
            file_1 = "ep_rewards_composition_%s_%r.npy" % (seed, self.goal_size)
            file_path = os.path.join(self.directory, file_1)
            np.save(file_path, self.episode_rewards_t)
            # file_0 = "ep_time_composition_%s_%r.npy" % (seed, self.goal_size)
            # file_path = os.path.join(self.directory, file_0)
            # np.save(file_path, self.episode_time)
            #file_2 = "ep_len_composition_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_2)
            #np.save(file_path, self.episode_length_t)
            #file_3 = "ep_res_composition_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_3)
            #np.save(file_path, self.resource_t)
            #file_4 = "ep_action_composition_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_4)
            #np.save(file_path, self.action_ls)

            # the Qtable array is saved in the file composition_qtable_seed.npy
            #file_5 = "composition_qtable_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_5)
            #np.save(file_path, self.Q_table)

            # Save the array to a text file
            #file_5_ = "q_learning_qtable_%s_%r.npy" % (seed, self.goal_size)
            #file_path = os.path.join(self.directory, file_5_)
            #np.savetxt(file_path, self.Q_table)

            #ls = []
            #for index, line in enumerate(self.Q_table):
             #   if np.any(line):
             #       print(line)
             #       print(index)
             #       ls.append(index)

            #print(len(ls))

            # save the state list
            #file_6 = "state_list.txt"
            #file_path = os.path.join(self.directory, file_6)
            #with open(file_path, 'w') as f:
             #   for s in self.state_ls:
              #      f.write(str(s) + '\n')

        self.episode_time = None
        self.episode_rewards_t = None
        self.episode_length_t = None
        self.resource_t = None
        self.action_ls = None
        self.Q_table = None
        gc.collect()  # Force garbage coll