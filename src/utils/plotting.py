""" git remote add origin https://github.com/Caston1984/Survival-Reinforcement-Learning.git
git branch -M main
git push -u origin main """

#python3 -m venv survival_rl  #my virtual environment
#source survival_rl/bin/activate #activate my virtual environment 
# pip3 install flask
# deactivate

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import pickle

def tuple_generator():
    # initializing N
    N = 22500

    # initializing Tuple element range
    R = 375

    # N Random Tuples list
    # Using list comprehension + sample()
    res = [divmod(ele, R + 1) for ele in random.sample(range((R + 1) * (R + 1)), N)]

    # printing result
    print("The N random tuples : " + str(res))

    # Find an element in list of tuples.
    Output = [item for item in res
              if item[0] == 4 or item[1] == 2]

    # printing output
    print('element', Output)

def snapshot():

    step = 2
    x = np.arange(0, 2000, step)
    goal_sizes = [10, 20, 30, 40, 50, 60, 70]
    seeds = [23, 51, 61, 94]
    directories = ['./trajectories/data2/exp_1', './trajectories/data2/exp_2', './trajectories/data2/exp_3', './trajectories/data2/exp_4', 
                   './trajectories/data2/exp_5', './trajectories/data2/exp_6', './trajectories/data2/exp_7', './trajectories/data2/exp_8', 
                   './trajectories/data2/exp_9', './trajectories/data2/exp_10', './trajectories/data2/exp_11', './trajectories/data2/exp_12',
                   './trajectories/data2/exp_13', './trajectories/data2/exp_14', './trajectories/data2/exp_15', './trajectories/data2/exp_16', 
                   './trajectories/data2/exp_17', './trajectories/data2/exp_18', './trajectories/data2/exp_19', './trajectories/data2/exp_20']
    a = {}
    b = {}
    c = {}
    # 20 experiments seedings, 7 goal sizes, 4 experiments seeds each with compositional rl, goal rl and q-learning rewards list of a 2000 episodes
    # This gives a overall reward matrix 'mat' of [20, 7, 4, 3, 1000] 
    experiment_ls = [] 
    for index, directory in enumerate(directories):                   
         
        goal_ls = [] 
        for goal_size in goal_sizes:                          
            seed_ls = []           
            for seed in seeds:                 
                reward_ls = []              
                file_1 = 'ep_rewards_composition_%s_%r.npy' % (seed, goal_size)
                file_path_1 = os.path.join(directory, file_1)                
                file_2 = 'ep_rewards_goal_%s_%r.npy' % (seed, goal_size)
                file_path_2 = os.path.join(directory, file_2)
                file_3 = 'ep_rewards_q_learning_%s_%r.npy' % (seed, goal_size)
                file_path_3 = os.path.join(directory, file_3)                
                comp_rl = np.load(file_path_1)
                goal_rl = np.load(file_path_2)
                qlearn_rl = np.load(file_path_3)

                #This code is there to smoothen the reward curve
                avg_ls_comp = []
                avg_ls_goal = []
                avg_ls_qlearn = []
                mean_ls_comp = []
                mean_ls_goal = []
                mean_ls_qlearn =[]

                for idx, reward in enumerate(comp_rl):
                    if (idx + 1) % step != 0:
                        mean_ls_comp.append(reward)
                    else:
                        avg_ls_comp.append(np.average(mean_ls_comp))
                a[seed] = avg_ls_comp

                for idx, reward in enumerate(goal_rl):
                    if (idx + 1) % step != 0:
                        mean_ls_goal.append(reward)
                    else:
                        avg_ls_goal.append(np.average(mean_ls_goal))
                b[seed] = avg_ls_goal

                for idx, reward in enumerate(qlearn_rl):
                    if (idx + 1) % step != 0:
                        mean_ls_qlearn.append(reward)
                    else:
                        avg_ls_qlearn.append(np.average(mean_ls_qlearn))
                c[seed] = avg_ls_qlearn
                
                reward_ls = [a[seed], b[seed], c[seed]]
                seed_ls.append(reward_ls)
            goal_ls.append(seed_ls)
        experiment_ls.append(goal_ls)
    
    # Save the list to a file
    with open("exp_list.pkl", "wb") as f:
       pickle.dump(experiment_ls, f)
           




def compare():
    # x-axis
    x = np.arange(0, 2000, 2)
            
    # Load the list from the file
    with open("exp_list.pkl", "rb") as f:
        experiment_ls = pickle.load(f)
    
    # Calculating the reward mean and standard deviation
    #[10, 20, 30, 40, 50, 60, 70]
    goal_size_c_10 = []
    goal_size_g_10 = []
    goal_size_q_10 = []     
    goal_size_c_20 = []
    goal_size_g_20 = []
    goal_size_q_20 = [] 
    goal_size_c_30 = []
    goal_size_g_30 = []
    goal_size_q_30 = [] 
    goal_size_c_40 = []
    goal_size_g_40 = []
    goal_size_q_40 = [] 
    goal_size_c_50 = []
    goal_size_g_50 = []
    goal_size_q_50 = []
    goal_size_c_60 = []
    goal_size_g_60 = []
    goal_size_q_60 = []
    goal_size_c_70 = []
    goal_size_g_70 = []
    goal_size_q_70 = [] 
    
    # Calculating the steps taken and and the interquartile range
    #[10, 20, 30, 40, 50, 60, 70]
    step_goal_size_c_10 = []
    step_goal_size_g_10 = []
    step_goal_size_q_10 = []     
    step_goal_size_c_20 = []
    step_goal_size_g_20 = []
    step_goal_size_q_20 = [] 
    step_goal_size_c_30 = []
    step_goal_size_g_30 = []
    step_goal_size_q_30 = [] 
    step_goal_size_c_40 = []
    step_goal_size_g_40 = []
    step_goal_size_q_40 = [] 
    step_goal_size_c_50 = []
    step_goal_size_g_50 = []
    step_goal_size_q_50 = []
    step_goal_size_c_60 = []
    step_goal_size_g_60 = []
    step_goal_size_q_60 = []
    step_goal_size_c_70 = []
    step_goal_size_g_70 = []
    step_goal_size_q_70 = [] 
            
    comp_ls = []
    goal_ls = []
    q_learning_ls = []   
    nr_exp = 20      
    goal_sizes = [10, 20, 30, 40, 50, 60, 70]
    seeds = [23, 51, 61, 94]
    
    #list to store the last numbers per goal size for all experiments
    x_10 = []
    x_20 = []
    x_30 = []
    x_40 = []
    x_50 = []
    x_60 = []
    x_70 = []
    b_10 = []
    b_20 = []
    b_30 = []
    b_40 = []
    b_50 = []
    b_60 = []
    b_70 = []
    c_10 = []
    c_20 = []
    c_30 = []
    c_40 = []
    c_50 = []
    c_60 = []
    c_70 = []
         
    for a in range(nr_exp):        
        for b, goal_size in enumerate(goal_sizes):  
            g_comp = [] 
            g_goal = [] 
            g_q_learning = []                         
            for c, seed in enumerate(seeds):
                comp = experiment_ls[a][b][c][0]
                goal = experiment_ls[a][b][c][1]
                q_learning = experiment_ls[a][b][c][2]
                # original step vs episode graphs for each experiment
                # converting the mean rewards graph into steps
                step_comp = [(x + 2)/0.001 for x in comp]
                step_goal = [(x + 2)//0.001 for x in goal]
                step_q_learning = [(x + 2)//0.001 for x in q_learning]
                
               # plt.boxplot(step_comp)
               # plt.show()
                """ title = 'Number of steps per episode, experiment %u, goal size %s, seed %r' %(a, goal_size, seed)
                # plotting the steps vs episode graph
                plt.plot(x, step_comp, label='Compositional RL')
                plt.plot(x, step_goal, label='Goal Oriented RL')
                plt.plot(x, step_q_learning, label= 'Q learning RL')
                plt.title(title)    
                plt.ylabel('Steps')
                plt.xlabel('Episode')
                plt.show()
                 """
                # Goal Size view stats
                g_comp.append(comp)
                g_goal.append(goal)
                g_q_learning.append(q_learning)
                # All goal size view stats
                comp_ls.append(comp)
                goal_ls.append(goal)
                q_learning_ls.append(q_learning)
                if goal_size == 10:
                    goal_size_c_10.append(experiment_ls[a][b][c][0])
                    goal_size_g_10.append(experiment_ls[a][b][c][1])
                    goal_size_q_10.append(experiment_ls[a][b][c][2])
                    x_10.append(comp[-1])
                    b_10.append(goal[-1])
                    c_10.append(q_learning[-1])
                    step_goal_size_c_10.append(step_comp)
                    step_goal_size_g_10.append(step_goal)
                    step_goal_size_q_10.append(step_q_learning)
                if goal_size == 20:
                    goal_size_c_20.append(experiment_ls[a][b][c][0])
                    goal_size_g_20.append(experiment_ls[a][b][c][1])
                    goal_size_q_20.append(experiment_ls[a][b][c][2])
                    x_20.append(comp[-1])
                    b_20.append(goal[-1])
                    c_20.append(q_learning[-1])
                    step_goal_size_c_20.append(step_comp)
                    step_goal_size_g_20.append(step_goal)
                    step_goal_size_q_20.append(step_q_learning)
                if goal_size == 30:
                    goal_size_c_30.append(experiment_ls[a][b][c][0])
                    goal_size_g_30.append(experiment_ls[a][b][c][1])
                    goal_size_q_30.append(experiment_ls[a][b][c][2])
                    x_30.append(comp[-1])
                    b_30.append(goal[-1])
                    c_30.append(q_learning[-1])
                    step_goal_size_c_30.append(step_comp)
                    step_goal_size_g_30.append(step_goal)
                    step_goal_size_q_30.append(step_q_learning)
                if goal_size == 40:
                    goal_size_c_40.append(experiment_ls[a][b][c][0])
                    goal_size_g_40.append(experiment_ls[a][b][c][1])
                    goal_size_q_40.append(experiment_ls[a][b][c][2])
                    x_40.append(comp[-1])
                    b_40.append(goal[-1])
                    c_40.append(q_learning[-1])
                    step_goal_size_c_40.append(step_comp)
                    step_goal_size_g_40.append(step_goal)
                    step_goal_size_q_40.append(step_q_learning)
                if goal_size == 50:
                    goal_size_c_50.append(experiment_ls[a][b][c][0])
                    goal_size_g_50.append(experiment_ls[a][b][c][1])
                    goal_size_q_50.append(experiment_ls[a][b][c][2])
                    x_50.append(comp[-1])
                    b_50.append(goal[-1])
                    c_50.append(q_learning[-1])
                    b_50.append(goal[-1])
                    c_50.append(q_learning[-1])
                    step_goal_size_c_50.append(step_comp)
                    step_goal_size_g_50.append(step_goal)
                    step_goal_size_q_50.append(step_q_learning)
                if goal_size == 60:
                    goal_size_c_60.append(experiment_ls[a][b][c][0])
                    goal_size_g_60.append(experiment_ls[a][b][c][1])
                    goal_size_q_60.append(experiment_ls[a][b][c][2])
                    x_60.append(comp[-1])
                    b_60.append(goal[-1])
                    c_60.append(q_learning[-1])
                    step_goal_size_c_60.append(step_comp)
                    step_goal_size_g_60.append(step_goal)
                    step_goal_size_q_60.append(step_q_learning)
                if goal_size == 70:
                    goal_size_c_70.append(experiment_ls[a][b][c][0])
                    goal_size_g_70.append(experiment_ls[a][b][c][1])
                    goal_size_q_70.append(experiment_ls[a][b][c][2])
                    x_70.append(comp[-1])
                    b_70.append(goal[-1])
                    c_70.append(q_learning[-1])
                    step_goal_size_c_70.append(step_comp)
                    step_goal_size_g_70.append(step_goal)
                    step_goal_size_q_70.append(step_q_learning)
            
             
            #title = 'Experiment %u: Reward curve for goal size %s' %(a, goal_size)
            #plt.plot(x, np.mean(g_comp, axis=0))
            #plt.fill_between(x, np.mean(g_comp, axis=0) - np.std(g_comp, axis=0),np.mean(g_comp, axis=0) + np.std(g_comp, axis=0), alpha=0.2)
            #plt.plot(x, np.mean(g_goal, axis=0))
            #plt.fill_between(x, np.mean(g_goal, axis=0) - np.std(g_goal, axis=0),np.mean(g_goal, axis=0) + np.std(g_goal, axis=0), alpha=0.2)
            #plt.plot(x, np.mean(g_q_learning, axis=0))
            #plt.fill_between(x, np.mean(g_q_learning, axis=0) - np.std(g_q_learning, axis=0),np.mean(g_q_learning, axis=0) + np.std(g_q_learning, axis=0), alpha=0.2)
            #plt.title(title)
            #plt.show()
            
    # Mean and Standard Deviation Plots per goal                
    goal_mean_c_10 = np.mean(goal_size_c_10, axis=0)        
    goal_std_c_10 = np.std(goal_size_c_10, axis=0)
    goal_mean_g_10 = np.mean(goal_size_g_10, axis=0)
    goal_std_g_10 = np.std(goal_size_g_10, axis=0)
    goal_mean_q_10 = np.mean(goal_size_q_10, axis=0)
    goal_std_q_10 = np.std(goal_size_q_10, axis=0)
            
    plt.plot(x, goal_mean_c_10, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_10-goal_std_c_10, goal_mean_c_10+goal_std_c_10, alpha=0.2)
    plt.plot(x, goal_mean_g_10, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_10-goal_std_g_10, goal_mean_g_10+goal_std_g_10, alpha=0.2)
    plt.plot(x, goal_mean_q_10, label='Q Learning')
    plt.fill_between(x, goal_mean_q_10-goal_std_q_10, goal_mean_q_10+goal_std_q_10, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 10 goal experiments')
    plt.show()
            
    goal_mean_c_20 = np.mean(goal_size_c_20, axis=0)        
    goal_std_c_20 = np.std(goal_size_c_20, axis=0)
    goal_mean_g_20 = np.mean(goal_size_g_20, axis=0)
    goal_std_g_20 = np.std(goal_size_g_20, axis=0)
    goal_mean_q_20 = np.mean(goal_size_q_20, axis=0)
    goal_std_q_20 = np.std(goal_size_q_20, axis=0)
            
    plt.plot(x, goal_mean_c_20, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_20-goal_std_c_20, goal_mean_c_20+goal_std_c_20, alpha=0.2)
    plt.plot(x, goal_mean_g_20, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_20-goal_std_g_20, goal_mean_g_20+goal_std_g_20, alpha=0.2)
    plt.plot(x, goal_mean_q_20, label='Q Learning')
    plt.fill_between(x, goal_mean_q_20-goal_std_q_20, goal_mean_q_20+goal_std_q_20, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 20 goal experiments')
    plt.show()
            
    goal_mean_c_30 = np.mean(goal_size_c_30, axis=0)        
    goal_std_c_30 = np.std(goal_size_c_30, axis=0)
    goal_mean_g_30 = np.mean(goal_size_g_30, axis=0)
    goal_std_g_30 = np.std(goal_size_g_30, axis=0)
    goal_mean_q_30 = np.mean(goal_size_q_30, axis=0)
    goal_std_q_30 = np.std(goal_size_q_30, axis=0)
            
    plt.plot(x, goal_mean_c_30, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_30-goal_std_c_30, goal_mean_c_30+goal_std_c_30, alpha=0.2)
    plt.plot(x, goal_mean_g_30, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_30-goal_std_g_30, goal_mean_g_30+goal_std_g_30, alpha=0.2)
    plt.plot(x, goal_mean_q_30, label='Q Learning')
    plt.fill_between(x, goal_mean_q_30-goal_std_q_30, goal_mean_q_30+goal_std_q_30, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 30 goal experiments')
    plt.show()
            
    goal_mean_c_40 = np.mean(goal_size_c_40, axis=0)        
    goal_std_c_40 = np.std(goal_size_c_40, axis=0)
    goal_mean_g_40 = np.mean(goal_size_g_40, axis=0)
    goal_std_g_40 = np.std(goal_size_g_40, axis=0)
    goal_mean_q_40 = np.mean(goal_size_q_40, axis=0)
    goal_std_q_40 = np.std(goal_size_q_40, axis=0)
            
    plt.plot(x, goal_mean_c_40, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_40-goal_std_c_40, goal_mean_c_40+goal_std_c_40, alpha=0.2)
    plt.plot(x, goal_mean_g_40, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_40-goal_std_g_40, goal_mean_g_40+goal_std_g_40, alpha=0.2)
    plt.plot(x, goal_mean_q_40, label='Q Learning')
    plt.fill_between(x, goal_mean_q_40-goal_std_q_40, goal_mean_q_40+goal_std_q_40, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 40 goal experiments')
    plt.show()
            
    goal_mean_c_50 = np.mean(goal_size_c_50, axis=0)        
    goal_std_c_50 = np.std(goal_size_c_50, axis=0)
    goal_mean_g_50 = np.mean(goal_size_g_50, axis=0)
    goal_std_g_50 = np.std(goal_size_g_50, axis=0)
    goal_mean_q_50 = np.mean(goal_size_q_50, axis=0)
    goal_std_q_50 = np.std(goal_size_q_50, axis=0)
            
    plt.plot(x, goal_mean_c_50, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_50-goal_std_c_50, goal_mean_c_50+goal_std_c_50, alpha=0.2)
    plt.plot(x, goal_mean_g_50, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_50-goal_std_g_50, goal_mean_g_50+goal_std_g_50, alpha=0.2)
    plt.plot(x, goal_mean_q_50, label='Q Learning')
    plt.fill_between(x, goal_mean_q_50-goal_std_q_50, goal_mean_q_50+goal_std_q_50, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 50 goal experiments')
    plt.show()
            
    goal_mean_c_60 = np.mean(goal_size_c_60, axis=0)        
    goal_std_c_60 = np.std(goal_size_c_60, axis=0)
    goal_mean_g_60 = np.mean(goal_size_g_60, axis=0)
    goal_std_g_60 = np.std(goal_size_g_60, axis=0)
    goal_mean_q_60 = np.mean(goal_size_q_60, axis=0)
    goal_std_q_60 = np.std(goal_size_q_60, axis=0)
            
    plt.plot(x, goal_mean_c_60, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_60-goal_std_c_60, goal_mean_c_60+goal_std_c_60, alpha=0.2)
    plt.plot(x, goal_mean_g_60, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_60-goal_std_g_60, goal_mean_g_60+goal_std_g_60, alpha=0.2)
    plt.plot(x, goal_mean_q_60, label='Q Learning')
    plt.fill_between(x, goal_mean_q_60-goal_std_q_60, goal_mean_q_60+goal_std_q_60, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 60 goal experiments')
    plt.show()
            
    goal_mean_c_70 = np.mean(goal_size_c_70, axis=0)        
    goal_std_c_70 = np.std(goal_size_c_70, axis=0)
    goal_mean_g_70 = np.mean(goal_size_g_70, axis=0)
    goal_std_g_70 = np.std(goal_size_g_70, axis=0)
    goal_mean_q_70 = np.mean(goal_size_q_70, axis=0)
    goal_std_q_70 = np.std(goal_size_q_70, axis=0)
            
    plt.plot(x, goal_mean_c_70, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_70-goal_std_c_70, goal_mean_c_70+goal_std_c_70, alpha=0.2)
    plt.plot(x, goal_mean_g_70, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_70-goal_std_g_70, goal_mean_g_70+goal_std_g_70, alpha=0.2)
    plt.plot(x, goal_mean_q_70, label='Q Learning')
    plt.fill_between(x, goal_mean_q_70-goal_std_q_70, goal_mean_q_70+goal_std_q_70, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 70 goal experiments')
    plt.show()
    
    # Step vs epsiode plots including inter-quartile range 
    step_goal_size_mean_c_10 = np.mean(step_goal_size_c_10, axis=0) 
    step_goal_size_mean_g_10 = np.mean(step_goal_size_g_10, axis=0)  
    step_goal_size_mean_q_10 = np.mean(step_goal_size_q_10, axis=0)   
    step_goal_size_mean_c_20 = np.mean(step_goal_size_c_20, axis=0) 
    step_goal_size_mean_g_20 = np.mean(step_goal_size_g_20, axis=0) 
    step_goal_size_mean_q_20 = np.mean(step_goal_size_q_20, axis=0) 
    step_goal_size_mean_c_30 = np.mean(step_goal_size_c_30, axis=0) 
    step_goal_size_mean_g_30 = np.mean(step_goal_size_g_30, axis=0) 
    step_goal_size_mean_q_30 = np.mean(step_goal_size_q_30, axis=0) 
    step_goal_size_mean_c_40 = np.mean(step_goal_size_c_40, axis=0) 
    step_goal_size_mean_g_40 = np.mean(step_goal_size_g_40, axis=0) 
    step_goal_size_mean_q_40 = np.mean(step_goal_size_q_40, axis=0) 
    step_goal_size_mean_c_50 = np.mean(step_goal_size_g_50, axis=0) 
    step_goal_size_mean_g_50 = np.mean(step_goal_size_c_50, axis=0) 
    step_goal_size_mean_q_50 = np.mean(step_goal_size_q_50, axis=0) 
    step_goal_size_mean_c_60 = np.mean(step_goal_size_c_60, axis=0) 
    step_goal_size_mean_g_60 = np.mean(step_goal_size_g_60, axis=0)
    step_goal_size_mean_q_60 = np.mean(step_goal_size_q_60, axis=0)
    step_goal_size_mean_c_70 = np.mean(step_goal_size_c_70, axis=0) 
    step_goal_size_mean_g_70 = np.mean(step_goal_size_g_70, axis=0) 
    step_goal_size_mean_q_70 = np.mean(step_goal_size_q_70, axis=0)              
    
    # https://www.datacamp.com/tutorial/python-boxplots
    data_group_1 = [step_goal_size_mean_c_10, step_goal_size_mean_g_10, step_goal_size_mean_q_10]
    data_group_2 = [step_goal_size_mean_c_20, step_goal_size_mean_g_20, step_goal_size_mean_q_20]
    data_group_3 = [step_goal_size_mean_c_30, step_goal_size_mean_g_30, step_goal_size_mean_q_30]
    data_group_4 = [step_goal_size_mean_c_40, step_goal_size_mean_g_40, step_goal_size_mean_q_40]
    data_group_5 = [step_goal_size_mean_c_50, step_goal_size_mean_g_50, step_goal_size_mean_q_50]
    data_group_6 = [step_goal_size_mean_c_60, step_goal_size_mean_g_60, step_goal_size_mean_q_60]
    data_group_7 = [step_goal_size_mean_c_70, step_goal_size_mean_g_70, step_goal_size_mean_q_70]
    data =  data_group_1 + data_group_2 + data_group_3 + data_group_4 + data_group_5 + data_group_6 + data_group_7    
    plt.boxplot(data, labels=['10-C', '10-G', '10-Q', '20-C', '20-G', '20-Q','30-C', '30-G', '30-Q', '40-C', '40-G', '40-Q', 
                              '50-C', '50-G', '50-Q', '60-C', '60-G', '60-Q', '70-C', '70-G', '70-Q'])
    plt.ylabel('Steps') 
    plt.xlabel('Goal size - Method')
    plt.title('Quartile Range Plot for all goal sizes')
    plt.show()
            
    
    # Plotting the rewards vs episode graph            
    mean_c = np.mean(comp_ls, axis=0)        
    std_c = np.std(comp_ls, axis=0)
    mean_g = np.mean(goal_ls, axis=0)
    std_g = np.std(goal_ls, axis=0)
    mean_q = np.mean(q_learning_ls, axis=0)
    std_q = np.std(q_learning_ls, axis=0)
       
        
    plt.plot(x, mean_c, label='Compositional RL')
    plt.fill_between(x, mean_c -std_c, mean_c + std_c, alpha=0.2)
    plt.plot(x, mean_g, label='Goal Oriented RL')
    plt.fill_between(x, mean_g - std_g, mean_g + std_g, alpha=0.2)
    plt.plot(x, mean_q, label='Q Learning')
    plt.fill_between(x, mean_q - std_q, mean_q + std_q, alpha=0.2)
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all experiments')
    plt.show()
    
    # plottting the rewards versus interquartile range per method
    data = [mean_c, mean_g, mean_q]
    plt.boxplot(data, labels=['Comp RL', 'Goal RL', 'Q learning RL'])
    plt.legend()
    plt.ylabel('Rewards')    
    plt.title('Rewards interquartile range')
    plt.show()
    
    # converting the mean rewards graph into steps
    step_mean_c = [(x + 2)//0.001 for x in mean_c]
    step_mean_g = [(x + 2)//0.001 for x in mean_g]
    step_mean_q = [(x + 2)//0.001 for x in mean_q]
    
    # plotting the steps vs episode graph
    plt.plot(x, step_mean_c, label='Compositional RL')
    plt.plot(x, step_mean_g, label='Goal Oriented RL')
    plt.plot(x, step_mean_q, label= 'Q learning RL')
    plt.legend()
    plt.title('Average number of steps per episode')    
    plt.ylabel('Steps')
    plt.xlabel('Episode')
    plt.show()
    
    
    # shift + ctrl + A for multiline comment
    variables = ["Comp RL", "Goal RL", "Q Learning RL"]
    d_1 = [x_10, x_20, x_30, x_40, x_50, x_60, x_70]
    d_2 = [b_10, b_20, b_30, b_40, b_50, b_60, b_70]
    d_1 = [c_10, c_20, c_30, c_40, c_50, c_60, c_70]
    x = ['10', '20', '30', '40', '50', '60', '70']
    
    mean_1 = [np.mean(x_10), np.mean(x_20), np.mean(x_30), np.mean(x_40), np.mean(x_50), np.mean(x_60), np.mean(x_70)]
    mean_2 = [np.mean(b_10), np.mean(b_20), np.mean(b_30), np.mean(b_40), np.mean(b_50), np.mean(b_60), np.mean(b_70)]
    mean_3 = [np.mean(c_10), np.mean(c_20), np.mean(c_30), np.mean(c_40), np.mean(c_50), np.mean(c_60), np.mean(c_70)]
    
    std_1 = [np.std(x_10), np.std(x_20), np.std(x_30), np.std(x_40), np.std(x_50), np.std(x_60), np.std(x_70)]
    std_2 = [np.std(b_10), np.std(b_20), np.std(b_30), np.std(b_40), np.std(b_50), np.std(b_60), np.std(b_70)]
    std_3 = [np.std(c_10), np.std(c_20), np.std(c_30), np.std(c_40), np.std(c_50), np.std(c_60), np.std(c_70)]
    plt.errorbar(x, mean_1, yerr=std_1, fmt='o', capsize=5, label='Comp RL Mean with Std Dev')
    plt.errorbar(x, mean_2, yerr=std_2, fmt='o', capsize=5, label='Goal RL Mean with Std Dev')
    plt.errorbar(x, mean_3, yerr=std_3, fmt='o', capsize=5, label='Q Learning RL Mean with Std Dev')
    plt.legend()
    #plt.plot(x, mean_1)
    #plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, alpha=0.2)
    #plt.plot(x, mean_2)
    #plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, alpha=0.2)
    #plt.plot(x, mean_3)
    #plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Goal Sizes')
    plt.title('Mean and the standard deviation of the rewards per goal size')
    plt.show() 

      



if __name__ == "__main__":
    # tuple_generator()
    #snapshot()
    compare()
    

