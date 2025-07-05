# https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
# https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial
from Composition_Q_learning_Lib_v2 import Q_learn_composition
from Goal_Oriented_Q_Learning_Lib_v2 import Goal_Q_learning
from Q_Learning_Lib_v2 import Q_learning
import gc

num_runs = 2000

dic_1 = {'f': 18, 'w': 5, 'x': 2, 'y': 4, 'g1_x': 2, 'g1_y': 5, 'g2_x': 9, 'g2_y': 2, 'g3_x': 4, 'g3_y': 10, 'g4_x': 5,
         'g4_y': 4, 'g1_f': 2, 'g1_w': 19, 'g2_f': 1, 'g2_w': 0, 'g3_f': 17, 'g3_w': 1, 'g4_f': 0, 'g4_w': 1,
         'dir': './trajectories/data2/exp_1'}

dic_2 = {'f': 9, 'w': 7, 'x': 5, 'y': 6, 'g1_x': 1, 'g1_y': 1, 'g2_x': 4, 'g2_y': 8, 'g3_x': 10, 'g3_y': 10, 'g4_x': 9,
         'g4_y': 2, 'g1_f': 3, 'g1_w': 1, 'g2_f': 2, 'g2_w': 5, 'g3_f': 2, 'g3_w': 1, 'g4_f': 0, 'g4_w': 1,
         'dir': './trajectories/data2/exp_2'}

dic_3 = {'f': 8, 'w': 6, 'x': 4, 'y': 6, 'g1_x': 3, 'g1_y': 1, 'g2_x': 4, 'g2_y': 3, 'g3_x': 8, 'g3_y': 6, 'g4_x': 9,
         'g4_y': 10, 'g1_f': 15, 'g1_w': 6, 'g2_f': 5, 'g2_w': 2, 'g3_f': 7, 'g3_w': 10, 'g4_f': 2, 'g4_w': 2,
         'dir': './trajectories/data2/exp_3'}

dic_4 = {'f': 14, 'w': 10, 'x': 4, 'y': 5, 'g1_x': 1, 'g1_y': 1, 'g2_x': 7, 'g2_y': 3, 'g3_x': 5, 'g3_y': 8, 'g4_x': 9,
         'g4_y': 10, 'g1_f': 12, 'g1_w': 17, 'g2_f': 6, 'g2_w': 6, 'g3_f': 9, 'g3_w': 6, 'g4_f': 8, 'g4_w': 12,
         'dir': './trajectories/data2/exp_4'}

dic_5 = {'f': 10, 'w': 10, 'x': 5, 'y': 5, 'g1_x': 3, 'g1_y': 3, 'g2_x': 9, 'g2_y': 3, 'g3_x': 3, 'g3_y': 9, 'g4_x': 9,
         'g4_y': 9, 'g1_f': 20, 'g1_w': 10, 'g2_f': 12, 'g2_w': 10, 'g3_f': 6, 'g3_w': 10, 'g4_f': 8, 'g4_w': 18,
         'dir': './trajectories/data2/exp_5'}

diktionary = [dic_1]

for dic in diktionary:
    (f, w, x, y, g1_x, g1_y, g2_x, g2_y, g3_x, g3_y, g4_x, g4_y, g1_f, g1_w, g2_f, g2_w, g3_f, g3_w, g4_f, g4_w, path) \
        = dic.values()

folders = ['./trajectories/data2/exp_10', './trajectories/data2/exp_11', './trajectories/data2/exp_12',
           './trajectories/data2/exp_13', './trajectories/data2/exp_14', './trajectories/data2/exp_15',
           './trajectories/data2/exp_16', './trajectories/data2/exp_17', './trajectories/data2/exp_18',
           './trajectories/data2/exp_19', './trajectories/data2/exp_20']
#'./trajectories/data2/exp_1', './trajectories/data2/exp_2', './trajectories/data2/exp_3',
 #          './trajectories/data2/exp_4', './trajectories/data2/exp_5', './trajectories/data2/exp_6',
  #         './trajectories/data2/exp_7', './trajectories/data2/exp_8', './trajectories/data2/exp_9',

#2, 13, 27, 49, 101, 115, 213, 227, 349,
env_seeds = [411, 5, 21, 501, 34, 17, 75, 82, 151, 64, 59]
goal_sizes = [7, 16, 32, 47, 53, 61, 76]

for env_seed, path in zip(env_seeds, folders):
    for goal_size in goal_sizes:
        print('Env seed', env_seed, 'GOAL SIZE', goal_size, 'Folder being saved', path)
        # Q-learning
        q_agent = Q_learning(env_seed, goal_size, path, n_training_episodes=num_runs)
        q_agent.initialize_q_table()
        q_agent.train()

        # Goal-Oriented RL
        goal_rl_agent = Goal_Q_learning(env_seed, goal_size, path, n_training_episodes=num_runs)
        goal_rl_agent.initialize_q_table()
        goal_rl_agent.train()

        # Composition RL
        base_goal_size = (goal_size // 2) + 1
        comp_rl_agent = Q_learn_composition(env_seed, goal_size, base_goal_size, path, n_training_episodes=num_runs)
        comp_rl_agent.initialize_q_table()
        comp_rl_agent.train()
        gc.collect()  # Force garbage collection




