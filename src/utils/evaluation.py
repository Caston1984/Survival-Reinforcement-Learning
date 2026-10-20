import imageio
from Gridworld import GridWorld
from Library import *
import random
import os

class Evaluation():
    def __init__(self, seed, goal_size, base_goals_size, directory):
        # random seed
        self.seed = seed
        random.seed(self.seed)

        # data directory
        self.directory = directory
        self.frame = []
        self.policy_ls = []
        self.episode_rewards_t = []

        # Environment parameters
        self.goal_size = goal_size
        self.base_goal_size = base_goals_size
        self.all_goal_pos = random.sample(self.goal_init(), goal_size)
        print('All goals', self.all_goal_pos)
        self.loaded_goals = self.all_goal_pos
        print('Loaded goals', self.loaded_goals)
        self.goals = [[pos, pos] for pos in self.all_goal_pos]
        self.start_pos = random.choice(self.goal_init())
        print('Start position', self.start_pos)
        self.start_state = [random.randint(20, 35), random.randint(20, 35), self.start_pos[0], self.start_pos[1]]
        print('Start state', self.start_state)
        self.goal_resources = self.resource_levels()
        print('Rewards', self.goal_resources)
        self.base_1, self.base_2 = self.base_constructor()
        #self.base_1 = [(1, 1), (1, 10)]  #random.sample(self.all_goal_pos, base_goals_size)
        #self.base_2 = [(1, 1), (10, 10)]  #random.sample(self.all_goal_pos, base_goals_size)

        self.T_states = [[pos, pos] for pos in self.all_goal_pos]

        # Seeds
        self.seeds = [5] #[5, 23, 51, 61, 94]

        # Load the state list
        self.state_ls = self.states()
        self.states = self.states()


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

    def base_constructor(self):
        base_1 = random.choices(self.loaded_goals, k=self.base_goal_size)
        base_2 = random.choices(self.loaded_goals, k=self.base_goal_size)
        return base_1, base_2

    def boolean_exp(self, a1=False, a2=False, a3=False, a4=False, a5=False, a6=False, a7=False, a8=False, a9=False,
                    a10=False, a11=False, a12=False, a13=False, a14=False):
        ls = []
        if a1:
            # AND(A, B)
            ls = list(set(self.base_1) & set(self.base_2))
        elif a2:
            # AND(A, NEG(B))
            ls =  list(set(self.base_1) - set(self.base_2))
        elif a3:
            # AND(NEG(A), B)
            ls = list(set(self.base_2) - set(self.base_1))
        elif a4:
            # NEG(OR(A, B))
            u = set(self.base_1) | set(self.base_2)
            ls = list(set(self.all_goal_pos) - u)
        elif a5:
            # A
            ls = list(set(self.base_1))
        elif a6:
            # NEG(A)
            ls = list(set(self.all_goal_pos) - set(self.base_1))
        elif a7:
            # B
            ls = list(set(self.base_2))
        elif a8:
            # NEG(B)
            ls = list(set(self.all_goal_pos) - set(self.base_2))
        elif a9:
            # OR(A, B)
            ls = list(set(self.base_1) | set(self.base_2))
        elif a10:
            # OR(A, NEG(B))
            ls = list(set(self.all_goal_pos) - set(self.base_2))
        elif a11:
            # OR(B, NEG(A))
            ls = list(set(self.all_goal_pos) - set(self.base_1))
        elif a12:
            # NEG(AND(A, B))
            y = set(self.base_1) & set(self.base_2)
            ls = list(set(self.all_goal_pos) - y)
        elif a13:
            # NEG(XOR(A, B))
            z = set(self.base_1) ^ set(self.base_2)
            ls = list(set(self.all_goal_pos) - z)
        elif a14:
            # XOR(A, B)
            ls = list(set(self.base_1) ^ set(self.base_2))
        return ls

    # function to check if the agent is alive
    def alive(self, arr):
        if ((arr[0] > 0) and (arr[0] < 40)) and ((arr[1] > 0) and (arr[1] < 42)):
            return True
        else:
            return False

    # return the index of the largest value in the supplied list
    # - arbitrarily select between the largest values in the case of a tie
    # (the standard np.argmax just chooses the first value in the case of a tie)
    def random_argmax(self, value_list):
        """ a random tie-breaking argmax """
        values = np.asarray(value_list)
        return np.argmax(np.random.random(values.shape) * (values == values.max()))

    # function to generate the state space
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
        maxiter = 2000
        #Bases = [self.base_1, self.base_2]
        self.tasks = [self.boolean_exp(a1=True), self.boolean_exp(a2=True), self.boolean_exp(a3=True),
                      self.boolean_exp(a4=True), self.boolean_exp(a5=True), self.boolean_exp(a6=True),
                      self.boolean_exp(a7=True), self.boolean_exp(a8=True), self.boolean_exp(a9=True),
                      self.boolean_exp(a10=True), self.boolean_exp(a11=True), self.boolean_exp(a12=True),
                      self.boolean_exp(a13=True), self.boolean_exp(a14=True)]

        # Learning universal bounds (min and max tasks)
        print('Learning the max task')
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=self.T_states, dense_rewards=False)
        EQ_max, stats, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)

        print('Learning the min task')
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=self.T_states, dense_rewards=False)
        EQ_min, stats, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter)

        # Learning base tasks and doing composed tasks
        print('Learning the first base task')
        goals = self.base_1
        goals = [[pos, pos] for pos in goals]
        print('Base 1 goals', goals)
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=goals, dense_rewards=False, T_states=self.T_states)
        A, stats1, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=self.T_states)

        print('Learning the second base task')
        goals = self.base_2
        goals = [[pos, pos] for pos in goals]
        print('Base 2 goals', goals)
        env = GridWorld(goal_reward=2, step_reward=-0.1, goals=goals, dense_rewards=False, T_states=self.T_states)
        B, stats2, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter, T_states=self.T_states)

        return EQ_max, EQ_min, A, B

    def goal_oriented_q_l(self):
        maxiter = 1000

        self.T_states = [[pos, pos] for pos in self.all_goal_pos]

        # Bases = [[self.goal_pos[0]], [self.goal_pos[1]], [self.goal_pos[2]], [self.goal_pos[3]]]

        # Learning the q_learning for the goals
        for goal in self.T_states:
            goal = [goal]
            env = GridWorld(goal_reward=2, step_reward=-0.1, goals=goal, dense_rewards=False,
                            T_states=self.T_states)  # , T_states=self.T_states
            A, stats1, time_ls1 = Goal_Oriented_Q_learning(env, maxiter=maxiter,
                                                           T_states=goal)  # , T_states=self.T_states

            self.policy_ls.append(A)
            print('Training of Goal complete')

        return

    def evaluate(self, goals, EQ, slip_prob=0):
        env = GridWorld(goals=goals, start_position=self.start_position, slip_prob=slip_prob)
        policy = EQ_P(EQ)
        grid_state = env.reset()
        done = False
        t = 0
        reward_ = 0

        while not done and self.alive(self.resources):
            action = policy[grid_state]
            grid_state, step_reward, done, _ = env.step(action)
            # rendering the video
            img = env.render()
            env.close()
            # Convert the canvas to a raw RGB buffer
            buf = img.canvas.tostring_rgb()  # img.canvas.buffer_rgba()
            ncols, nrows = img.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            self.frame.append(image)

            self.resources = np.subtract(self.resources, np.array([1, 1]))
            self.step += 1
            reward_ = reward_ + step_reward
            t += 1

        if done:
            idx = self.all_goal_pos.index(env.position)
            self.resources = np.add(self.resources, np.array(self.goal_resources[idx]))

        alive = self.alive(self.resources)

        if not alive:
            reward_ = reward_ - 1

        return reward_, env.position

    def evaluate_2(self, goals, EQ, slip_prob=0):
        env = GridWorld(goals=goals, start_position=self.start_position, slip_prob=slip_prob)
        policy = EQ_P(EQ)
        grid_state = env.reset()
        done = False
        t = 0
        reward = 0

        while not done and self.alive(self.resources):
            action = policy[grid_state]
            grid_state, step_reward, done, _ = env.step(action)
            # rendering the video
            img = env.render()
            env.close()
            # Convert the canvas to a raw RGB buffer
            buf = img.canvas.tostring_rgb() #img.canvas.buffer_rgba()
            ncols, nrows = img.canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            self.frame.append(image)

            self.resources = np.subtract(self.resources, np.array([1, 1]))
            self.step += 1
            reward = reward + step_reward
            t += 1

        if done:
            if env.position == self.goal_pos[0]:
                self.resources = np.add(self.resources, np.array(self.goal_res[0]))
            elif env.position == self.goal_pos[1]:
                self.resources = np.add(self.resources, np.array(self.goal_res[1]))
            elif env.position == self.goal_pos[2]:
                self.resources = np.add(self.resources, np.array(self.goal_res[2]))
            elif env.position == self.goal_pos[3]:
                self.resources = np.add(self.resources, np.array(self.goal_res[3]))

        if not self.alive(self.resources):
            reward = reward - 10

        return reward, env.position

    def evaluate_comp_agent(self):

        # # Load the actions which are the composed tasks and their respective goals
        EQ_max, EQ_min, A, B = self.composition()
        NEG = lambda x: NOT(x, EQ_max=EQ_max, EQ_min=EQ_min)
        XOR = lambda EQ1, EQ2: OR(AND(EQ1, NEG(EQ2)), AND(EQ2, NEG(EQ1)))
        self.composed = [AND(A, B), AND(A, NEG(B)), AND(B, NEG(A)), NEG(OR(A, B)), A, NEG(A), B, NEG(B), OR(A, B),
                          OR(A, NEG(B)), OR(B, NEG(A)), NEG(AND(A, B)), NEG(XOR(A, B)), XOR(A, B)]

        print('composition rl agent')
        for seed in self.seeds:
            flag = True
            while flag:
                print('seed selected', seed)
                random.seed(seed)

                file_1 = "composition_qtable_%s_%r.npy" % (seed, self.goal_size)
                file_path_1 = os.path.join(self.directory, file_1)
                self.q_table = np.load(file_path_1)
                self.frame = []
                # Because we want to evaluate if the agent has learnt to leave forver, we only evaluate one episode and see how long it takes
                total_rewards_ep = 0
                self.step = 0
                self.state = self.start_state
                self.start_position = (self.state[2], self.state[3])
                self.resources = [self.state[0], self.state[1]]

                while self.alive(self.resources):
                    self.step += 1
                    state_id = self.states.index(self.state)
                    action = self.random_argmax(self.q_table[state_id])

                    # the selected action is the index for the composition to be executed
                    goals = [[pos, pos] for pos in self.tasks[action]]

                    # Evaluate the composition policy selected
                    reward, position = self.evaluate(goals, self.composed[action])
                    total_rewards_ep += reward

                    new_position = position
                    new_state = [self.resources[0], self.resources[1], new_position[0], new_position[1]]

                    # Assign the new state as the current
                    self.state = new_state

                    # If alive current position is start position
                    alive = self.alive(self.resources)
                    if alive:
                        self.start_position = position

                    if total_rewards_ep > 0.012:
                        flag = False
                        break

                    #if not alive:
                     #   flag = False



                    print('Total_rewards', total_rewards_ep)

            print('Final Total Rewards', total_rewards_ep)

            file_3 = "composition_video_%s.mp4" % (seed)
            file_path = os.path.join(self.directory, file_3)
            imageio.mimsave(file_path, self.frame, fps=10)

    def evaluate_goal_rl_agent(self):

        print('goal rl agent')

        # the learnt q_tables of the above goals
        self.goal_oriented_q_l()

        # list of the q_tables
        #goal_q_tables = [a, b, c, d]

        # Load the numpy file with the Q-table for any random seed used
        for seed in self.seeds:
            flag = True
            while flag:
                print('seed selected', seed)
                random.seed(seed)

                file_1 = "gorl_qtable_%s_%r.npy" % (seed, self.goal_size)
                file_path_1 = os.path.join(self.directory, file_1)
                self.q_table = np.load(file_path_1)
                self.frame = []
                # Because we want to evaluate if the agent has learnt to leave forver, we only evaluate one episode and see how long it takes
                total_rewards_ep = 0
                self.step = 0
                self.state = self.start_state
                self.start_position = (self.state[2], self.state[3])
                self.resources = [self.state[0], self.state[1]]

                goals = [[pos, pos] for pos in self.all_goal_pos]

                while self.alive(self.resources):
                    self.step += 1
                    state_id = self.state_ls.index(self.state)
                    action = self.random_argmax(self.q_table[state_id])
                    print('In state', state_id, 'the q_table record is', self.q_table[state_id])
                    # the selected action is the index for the composition to be executed
                    goal = goals[action]
                    goal = [[pos, pos] for pos in goal]
                    print('action', action, 'goal',goal)

                    policy = self.policy_ls[action]

                    # Evaluate the composition policy selected
                    reward, position = self.evaluate(goal, policy)
                    total_rewards_ep += reward

                    new_position = position
                    new_state = [self.resources[0], self.resources[1], new_position[0], new_position[1]]

                    # Assign the new state as the current
                    self.state = new_state

                    # If alive current position is start position
                    alive = self.alive(self.resources)
                    if alive:
                        self.start_position = position

                    #if not alive:
                     #   flag = False

                    if total_rewards_ep > 0.025:
                        flag = False
                        break

                    print('Interim total', total_rewards_ep)
            print('Final Total', total_rewards_ep)
            file_3 = "goal_rl_video_%s.mp4" % (seed)
            file_path = os.path.join(self.directory, file_3)
            imageio.mimsave(file_path, self.frame, fps=10)


    def evaluate_q_learning_agent(self):

        goals = [[pos, pos] for pos in self.all_goal_pos]

        print(goals)

        for seed in self.seeds:
            print('seed selected', seed)
            flag = True
            while flag:
                random.seed(seed)
                file_1 = "q_learning_qtable_%s_%r.npy" % (seed, self.goal_size)
                file_path_1 = os.path.join(self.directory, file_1)
                self.q_table = np.load(file_path_1)
                self.frame = []
                # Because we want to evaluate if the agent has learnt to leave forver, we only evaluate one episode and see how long it takes
                total_rewards_ep = 0
                self.step = 0
                self.state = self.start_state
                self.start_position = (self.state[2], self.state[3])
                self.resources = [self.state[0], self.state[1]]
                env = GridWorld(goals=goals, start_position=self.start_position)
                state_id = self.state_ls.index(self.state)
                print('Qtable len', len(self.q_table))
                while self.alive(self.resources):
                    print('Q table entry', self.q_table[state_id])
                    action = self.random_argmax(self.q_table[state_id])
                    print('action', action)
                    state_, step_reward, done, _ = env.step(action)
                    # rendering the video
                    img = env.render()
                    env.close()
                    # Convert the canvas to a raw RGB buffer
                    buf = img.canvas.tostring_rgb() #img.canvas.buffer_rgba()
                    ncols, nrows = img.canvas.get_width_height()
                    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
                    self.frame.append(image)
                    total_rewards_ep += step_reward

                    self.resources = np.subtract(self.resources, np.array([1, 1]))
                    self.step += 1

                    if done:
                        idx = self.all_goal_pos.index(env.position)
                        self.resources = np.add(self.resources, np.array(self.goal_resources[idx]))

                        env = GridWorld(goals=goals, start_position=env.position)

                    alive = self.alive(self.resources)
                    if not alive:
                        total_rewards_ep = total_rewards_ep - 1
                    print('Current Total rewards', total_rewards_ep)
                    if total_rewards_ep >= 0.019:
                        flag = False
                        break
                    #self.step += 1
                    #if self.step > 100:
                    #   break

                    new_position = env.position
                    new_state = [self.resources[0], self.resources[1], new_position[0], new_position[1]]

                    new_state_id = self.state_ls.index(new_state)

                    # Our state is the new state
                    state_id = new_state_id

                print(total_rewards_ep)
            print('rendering')
            file_3 = "q_learning_rl_video_%s.mp4" % (seed)
            file_path = os.path.join(self.directory, file_3)
            imageio.mimsave(file_path, self.frame, fps=10)


# Instantiate an object
evaluate = Evaluation(411, 7, 4, './trajectories/data2/exp_10')

#evaluate = Evaluation(9, 7, 5, 6, 1, 1, 4, 8, 10, 10, 9, 2,
   #               3, 1, 2, 5, 2, 1, 0, 1, './trajectories/data/exp_2')

#evaluate = Evaluation(8, 6, 4, 6, 3, 1, 4, 3, 8, 6, 9, 10,
 #                 15, 6, 5, 2, 7, 10, 2, 2, './trajectories/data/exp_3')

#evaluate = Evaluation(14,10,4,5,1,1,7,3,5,8,9,10,12,17,
  #                6,6,9,6,8,12, './trajectories/data/exp_4')

#evaluate = Evaluation(10, 10, 5, 5, 3, 3, 9, 3, 3, 9, 9, 9,
 #                     20, 10, 12, 10, 6, 10, 8, 18, './trajectories/data/exp_5')

# Evaluate the agent
#evaluate.evaluate_q_learning_agent()
#evaluate.evaluate_goal_rl_agent()
evaluate.evaluate_comp_agent()

