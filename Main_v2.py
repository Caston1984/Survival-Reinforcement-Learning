# https://kfoofw.github.io/contextual-bandits-linear-ucb-disjoint/
# https://www.datacamp.com/tutorial/introduction-q-learning-beginner-tutorial
from Composition_Q_learning_Lib_v2 import Q_learn_composition
from Goal_Oriented_Q_Learning_Lib_v2 import Goal_Q_learning
from Q_Learning_Lib_v2 import Q_learning
import gc

num_runs = 2000


folders = ['./trajectories/data2/exp_1', './trajectories/data2/exp_2', './trajectories/data2/exp_3',
            './trajectories/data2/exp_4', './trajectories/data2/exp_5', './trajectories/data2/exp_6',
            './trajectories/data2/exp_7', './trajectories/data2/exp_8', './trajectories/data2/exp_9',
            './trajectories/data2/exp_10', './trajectories/data2/exp_11', './trajectories/data2/exp_12',
            './trajectories/data2/exp_13', './trajectories/data2/exp_14', './trajectories/data2/exp_15',
            './trajectories/data2/exp_16', './trajectories/data2/exp_17', './trajectories/data2/exp_18',
            './trajectories/data2/exp_19', './trajectories/data2/exp_20']
    

env_seeds = [2, 13, 27, 49, 101, 115, 213, 227, 349, 411, 5, 21, 501, 34, 17, 75, 82, 151, 64, 59]
goal_sizes = [10, 20, 30, 40, 50, 60, 70]

for env_seed, path in zip(env_seeds, folders):
    for goal_size in goal_sizes:
        print('Env seed', env_seed, 'GOAL SIZE', goal_size, 'Folder being saved', path)
        # Goal-Oriented RL
        goal_rl_agent = Goal_Q_learning(env_seed, goal_size, path, n_training_episodes=num_runs)
        goal_rl_agent.initialize_q_table()
        goal_rl_agent.train()
        gc.collect()  # Force garbage collection




