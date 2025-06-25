""" git remote add origin https://github.com/Caston1984/Survival-Reinforcement-Learning.git
git branch -M main
git push -u origin main """

#python3 -m venv survival_rl  #my virtual environment
#source survival_rl/bin/activate #activate my virtual environment 
# pip3 install flask
# deactivate

from cProfile import label
from logging.handlers import DEFAULT_SOAP_LOGGING_PORT
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
    goal_sizes = [7, 16, 32, 47, 53, 61, 76]
    seeds = [5, 23, 51, 61, 94]
    directories = ['./trajectories/data2/exp_1', './trajectories/data2/exp_2', './trajectories/data2/exp_3', './trajectories/data2/exp_4', 
                   './trajectories/data2/exp_5', './trajectories/data2/exp_6', './trajectories/data2/exp_7', './trajectories/data2/exp_8', 
                   './trajectories/data2/exp_9', './trajectories/data2/exp_10', './trajectories/data2/exp_11', './trajectories/data2/exp_12',
                   './trajectories/data2/exp_13', './trajectories/data2/exp_14', './trajectories/data2/exp_15', './trajectories/data2/exp_16', 
                   './trajectories/data2/exp_17', './trajectories/data2/exp_18', './trajectories/data2/exp_19', './trajectories/data2/exp_20']
    a = {}
    b = {}
    c = {}
    # 20 experiments seedings, 7 goal sizes, 5 experiments seeds each with compositional rl, goal rl and q-learning rewards list of a 1000 episodes
    # This gives a overall reward matrix 'mat' of [20, 7, 5, 3, 1000] 
    experiment_ls = [] 
    for index, directory in enumerate(directories):                    
        print('Exp', index)  
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
    #[7, 16, 32, 47, 53, 61, 76]
    goal_size_c_7 = []
    goal_size_g_7 = []
    goal_size_q_7 = []     
    goal_size_c_16 = []
    goal_size_g_16 = []
    goal_size_q_16 = [] 
    goal_size_c_32 = []
    goal_size_g_32 = []
    goal_size_q_32 = [] 
    goal_size_c_47 = []
    goal_size_g_47 = []
    goal_size_q_47 = [] 
    goal_size_c_53 = []
    goal_size_g_53 = []
    goal_size_q_53 = []
    goal_size_c_61 = []
    goal_size_g_61 = []
    goal_size_q_61 = []
    goal_size_c_76 = []
    goal_size_g_76 = []
    goal_size_q_76 = [] 
    
    # Calculating the steps taken and and the interquartile range
    #[7, 16, 32, 47, 53, 61, 76]
    step_goal_size_c_7 = []
    step_goal_size_g_7 = []
    step_goal_size_q_7 = []     
    step_goal_size_c_16 = []
    step_goal_size_g_16 = []
    step_goal_size_q_16 = [] 
    step_goal_size_c_32 = []
    step_goal_size_g_32 = []
    step_goal_size_q_32 = [] 
    step_goal_size_c_47 = []
    step_goal_size_g_47 = []
    step_goal_size_q_47 = [] 
    step_goal_size_c_53 = []
    step_goal_size_g_53 = []
    step_goal_size_q_53 = []
    step_goal_size_c_61 = []
    step_goal_size_g_61 = []
    step_goal_size_q_61 = []
    step_goal_size_c_76 = []
    step_goal_size_g_76 = []
    step_goal_size_q_76 = [] 
            
    comp_ls = []
    goal_ls = []
    q_learning_ls = []   
    nr_exp = 20      
    goal_sizes = [7, 16, 32, 47, 53, 61, 76]
    seeds = [5, 23, 51, 61, 94]
    
    #list to store the last numbers per goal size for all experiments
    x_7 = []
    x_16 = []
    x_32 = []
    x_47 = []
    x_53 = []
    x_61 = []
    x_76 = []
    b_7 = []
    b_16 = []
    b_32 = []
    b_47 = []
    b_53 = []
    b_61 = []
    b_76 = []
    c_7 = []
    c_16 = []
    c_32 = []
    c_47 = []
    c_53 = []
    c_61 = []
    c_76 = []
         
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
                
                # plotting the steps vs episode graph
                """ plt.plot(x, step_comp, label='Compositional RL')
                plt.plot(x, step_goal, label='Goal Oriented RL')
                plt.plot(x, step_q_learning, label= 'Q learning RL')
                plt.title('Number of steps per episode')    
                plt.ylabel('Steps')
                plt.xlabel('Episode')
                plt.show() """
                
                # Goal Size view stats
                g_comp.append(comp)
                g_goal.append(goal)
                g_q_learning.append(q_learning)
                # All goal size view stats
                comp_ls.append(comp)
                goal_ls.append(goal)
                q_learning_ls.append(q_learning)
                if goal_size == 7:
                    goal_size_c_7.append(experiment_ls[a][b][c][0])
                    goal_size_g_7.append(experiment_ls[a][b][c][1])
                    goal_size_q_7.append(experiment_ls[a][b][c][2])
                    x_7.append(comp[-1])
                    b_7.append(goal[-1])
                    c_7.append(q_learning[-1])
                    step_goal_size_c_7.append(step_comp)
                    step_goal_size_g_7.append(step_goal)
                    step_goal_size_q_7.append(step_q_learning)
                if goal_size == 16:
                    goal_size_c_16.append(experiment_ls[a][b][c][0])
                    goal_size_g_16.append(experiment_ls[a][b][c][1])
                    goal_size_q_16.append(experiment_ls[a][b][c][2])
                    x_16.append(comp[-1])
                    b_16.append(goal[-1])
                    c_16.append(q_learning[-1])
                    step_goal_size_c_16.append(step_comp)
                    step_goal_size_g_16.append(step_goal)
                    step_goal_size_q_16.append(step_q_learning)
                if goal_size == 32:
                    goal_size_c_32.append(experiment_ls[a][b][c][0])
                    goal_size_g_32.append(experiment_ls[a][b][c][1])
                    goal_size_q_32.append(experiment_ls[a][b][c][2])
                    x_32.append(comp[-1])
                    b_32.append(goal[-1])
                    c_32.append(q_learning[-1])
                    step_goal_size_c_32.append(step_comp)
                    step_goal_size_g_32.append(step_goal)
                    step_goal_size_q_32.append(step_q_learning)
                if goal_size == 47:
                    goal_size_c_47.append(experiment_ls[a][b][c][0])
                    goal_size_g_47.append(experiment_ls[a][b][c][1])
                    goal_size_q_47.append(experiment_ls[a][b][c][2])
                    x_47.append(comp[-1])
                    b_47.append(goal[-1])
                    c_47.append(q_learning[-1])
                    step_goal_size_c_47.append(step_comp)
                    step_goal_size_g_47.append(step_goal)
                    step_goal_size_q_47.append(step_q_learning)
                if goal_size == 53:
                    goal_size_c_53.append(experiment_ls[a][b][c][0])
                    goal_size_g_53.append(experiment_ls[a][b][c][1])
                    goal_size_q_53.append(experiment_ls[a][b][c][2])
                    x_53.append(comp[-1])
                    b_53.append(goal[-1])
                    c_53.append(q_learning[-1])
                    b_53.append(goal[-1])
                    c_53.append(q_learning[-1])
                    step_goal_size_c_53.append(step_comp)
                    step_goal_size_g_53.append(step_goal)
                    step_goal_size_q_53.append(step_q_learning)
                if goal_size == 61:
                    goal_size_c_61.append(experiment_ls[a][b][c][0])
                    goal_size_g_61.append(experiment_ls[a][b][c][1])
                    goal_size_q_61.append(experiment_ls[a][b][c][2])
                    x_61.append(comp[-1])
                    b_61.append(goal[-1])
                    c_61.append(q_learning[-1])
                    step_goal_size_c_61.append(step_comp)
                    step_goal_size_g_61.append(step_goal)
                    step_goal_size_q_61.append(step_q_learning)
                if goal_size == 76:
                    goal_size_c_76.append(experiment_ls[a][b][c][0])
                    goal_size_g_76.append(experiment_ls[a][b][c][1])
                    goal_size_q_76.append(experiment_ls[a][b][c][2])
                    x_76.append(comp[-1])
                    b_76.append(goal[-1])
                    c_76.append(q_learning[-1])
                    step_goal_size_c_76.append(step_comp)
                    step_goal_size_g_76.append(step_goal)
                    step_goal_size_q_76.append(step_q_learning)
            
             
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
    ''' goal_mean_c_7 = np.mean(goal_size_c_7, axis=0)        
    goal_std_c_7 = np.std(goal_size_c_7, axis=0)
    goal_mean_g_7 = np.mean(goal_size_g_7, axis=0)
    goal_std_g_7 = np.std(goal_size_g_7, axis=0)
    goal_mean_q_7 = np.mean(goal_size_q_7, axis=0)
    goal_std_q_7 = np.std(goal_size_q_7, axis=0)
            
    plt.plot(x, goal_mean_c_7, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_7-goal_std_c_7, goal_mean_c_7+goal_std_c_7, alpha=0.2)
    plt.plot(x, goal_mean_g_7, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_7-goal_std_g_7, goal_mean_g_7+goal_std_g_7, alpha=0.2)
    plt.plot(x, goal_mean_q_7, label='Q Learning')
    plt.fill_between(x, goal_mean_q_7-goal_std_q_7, goal_mean_q_7+goal_std_q_7, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 7 goal experiments')
    plt.show()
            
    goal_mean_c_16 = np.mean(goal_size_c_16, axis=0)        
    goal_std_c_16 = np.std(goal_size_c_16, axis=0)
    goal_mean_g_16 = np.mean(goal_size_g_16, axis=0)
    goal_std_g_16 = np.std(goal_size_g_16, axis=0)
    goal_mean_q_16 = np.mean(goal_size_q_16, axis=0)
    goal_std_q_16 = np.std(goal_size_q_16, axis=0)
            
    plt.plot(x, goal_mean_c_16, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_16-goal_std_c_16, goal_mean_c_16+goal_std_c_16, alpha=0.2)
    plt.plot(x, goal_mean_g_16, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_16-goal_std_g_16, goal_mean_g_16+goal_std_g_16, alpha=0.2)
    plt.plot(x, goal_mean_q_16, label='Q Learning')
    plt.fill_between(x, goal_mean_q_16-goal_std_q_16, goal_mean_q_16+goal_std_q_16, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 16 goal experiments')
    plt.show()
            
    goal_mean_c_32 = np.mean(goal_size_c_32, axis=0)        
    goal_std_c_32 = np.std(goal_size_c_32, axis=0)
    goal_mean_g_32 = np.mean(goal_size_g_32, axis=0)
    goal_std_g_32 = np.std(goal_size_g_32, axis=0)
    goal_mean_q_32 = np.mean(goal_size_q_32, axis=0)
    goal_std_q_32 = np.std(goal_size_q_32, axis=0)
            
    plt.plot(x, goal_mean_c_32, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_32-goal_std_c_32, goal_mean_c_32+goal_std_c_32, alpha=0.2)
    plt.plot(x, goal_mean_g_32, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_32-goal_std_g_32, goal_mean_g_32+goal_std_g_32, alpha=0.2)
    plt.plot(x, goal_mean_q_32, label='Q Learning')
    plt.fill_between(x, goal_mean_q_32-goal_std_q_32, goal_mean_q_32+goal_std_q_32, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 32 goal experiments')
    plt.show()
            
    goal_mean_c_47 = np.mean(goal_size_c_47, axis=0)        
    goal_std_c_47 = np.std(goal_size_c_47, axis=0)
    goal_mean_g_47 = np.mean(goal_size_g_47, axis=0)
    goal_std_g_47 = np.std(goal_size_g_47, axis=0)
    goal_mean_q_47 = np.mean(goal_size_q_47, axis=0)
    goal_std_q_47 = np.std(goal_size_q_47, axis=0)
            
    plt.plot(x, goal_mean_c_47, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_47-goal_std_c_47, goal_mean_c_47+goal_std_c_47, alpha=0.2)
    plt.plot(x, goal_mean_g_47, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_47-goal_std_g_47, goal_mean_g_47+goal_std_g_47, alpha=0.2)
    plt.plot(x, goal_mean_q_47, label='Q Learning')
    plt.fill_between(x, goal_mean_q_47-goal_std_q_47, goal_mean_q_47+goal_std_q_47, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 47 goal experiments')
    plt.show()
            
    goal_mean_c_53 = np.mean(goal_size_c_53, axis=0)        
    goal_std_c_53 = np.std(goal_size_c_53, axis=0)
    goal_mean_g_53 = np.mean(goal_size_g_53, axis=0)
    goal_std_g_53 = np.std(goal_size_g_53, axis=0)
    goal_mean_q_53 = np.mean(goal_size_q_53, axis=0)
    goal_std_q_53 = np.std(goal_size_q_53, axis=0)
            
    plt.plot(x, goal_mean_c_53, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_53-goal_std_c_53, goal_mean_c_53+goal_std_c_53, alpha=0.2)
    plt.plot(x, goal_mean_g_53, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_53-goal_std_g_53, goal_mean_g_53+goal_std_g_53, alpha=0.2)
    plt.plot(x, goal_mean_q_53, label='Q Learning')
    plt.fill_between(x, goal_mean_q_53-goal_std_q_53, goal_mean_q_53+goal_std_q_53, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 53 goal experiments')
    plt.show()
            
    goal_mean_c_61 = np.mean(goal_size_c_61, axis=0)        
    goal_std_c_61 = np.std(goal_size_c_61, axis=0)
    goal_mean_g_61 = np.mean(goal_size_g_61, axis=0)
    goal_std_g_61 = np.std(goal_size_g_61, axis=0)
    goal_mean_q_61 = np.mean(goal_size_q_61, axis=0)
    goal_std_q_61 = np.std(goal_size_q_61, axis=0)
            
    plt.plot(x, goal_mean_c_61, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_61-goal_std_c_61, goal_mean_c_61+goal_std_c_61, alpha=0.2)
    plt.plot(x, goal_mean_g_61, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_61-goal_std_g_61, goal_mean_g_61+goal_std_g_61, alpha=0.2)
    plt.plot(x, goal_mean_q_61, label='Q Learning')
    plt.fill_between(x, goal_mean_q_61-goal_std_q_61, goal_mean_q_61+goal_std_q_61, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 53 goal experiments')
    plt.show()
            
    goal_mean_c_76 = np.mean(goal_size_c_76, axis=0)        
    goal_std_c_76 = np.std(goal_size_c_76, axis=0)
    goal_mean_g_76 = np.mean(goal_size_g_76, axis=0)
    goal_std_g_76 = np.std(goal_size_g_76, axis=0)
    goal_mean_q_76 = np.mean(goal_size_q_76, axis=0)
    goal_std_q_76 = np.std(goal_size_q_76, axis=0)
            
    plt.plot(x, goal_mean_c_76, label='Compositional RL')
    plt.fill_between(x, goal_mean_c_76-goal_std_c_76, goal_mean_c_76+goal_std_c_76, alpha=0.2)
    plt.plot(x, goal_mean_g_76, label='Goal Oriented RL')
    plt.fill_between(x, goal_mean_g_76-goal_std_g_76, goal_mean_g_76+goal_std_g_76, alpha=0.2)
    plt.plot(x, goal_mean_q_76, label='Q Learning')
    plt.fill_between(x, goal_mean_q_76-goal_std_q_76, goal_mean_q_76+goal_std_q_76, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all 76 goal experiments')
    plt.show() '''
    
    # Step vs epsiode plots including inter-quartile range 
    step_goal_size_mean_c_7 = np.mean(step_goal_size_c_7, axis=0) 
    step_goal_size_mean_g_7 = np.mean(step_goal_size_g_7, axis=0)  
    step_goal_size_mean_q_7 = np.mean(step_goal_size_q_7, axis=0)   
    step_goal_size_mean_c_16 = np.mean(step_goal_size_c_16, axis=0) 
    step_goal_size_mean_g_16 = np.mean(step_goal_size_g_16, axis=0) 
    step_goal_size_mean_q_16 = np.mean(step_goal_size_q_16, axis=0) 
    step_goal_size_mean_c_32 = np.mean(step_goal_size_c_32, axis=0) 
    step_goal_size_mean_g_32 = np.mean(step_goal_size_g_32, axis=0) 
    step_goal_size_mean_q_32 = np.mean(step_goal_size_q_32, axis=0) 
    step_goal_size_mean_c_47 = np.mean(step_goal_size_c_47, axis=0) 
    step_goal_size_mean_g_47 = np.mean(step_goal_size_g_47, axis=0) 
    step_goal_size_mean_q_47 = np.mean(step_goal_size_q_47, axis=0) 
    step_goal_size_mean_c_53 = np.mean(step_goal_size_g_53, axis=0) 
    step_goal_size_mean_g_53 = np.mean(step_goal_size_c_53, axis=0) 
    step_goal_size_mean_q_53 = np.mean(step_goal_size_q_53, axis=0) 
    step_goal_size_mean_c_61 = np.mean(step_goal_size_c_61, axis=0) 
    step_goal_size_mean_g_61 = np.mean(step_goal_size_g_61, axis=0)
    step_goal_size_mean_q_61 = np.mean(step_goal_size_q_61, axis=0)
    step_goal_size_mean_c_76 = np.mean(step_goal_size_c_76, axis=0) 
    step_goal_size_mean_g_76 = np.mean(step_goal_size_g_76, axis=0) 
    step_goal_size_mean_q_76 = np.mean(step_goal_size_q_76, axis=0)              
    
    # https://www.datacamp.com/tutorial/python-boxplots
    data_group_1 = [step_goal_size_mean_c_7, step_goal_size_mean_g_7, step_goal_size_mean_q_7]
    data_group_2 = [step_goal_size_mean_c_16, step_goal_size_mean_g_16, step_goal_size_mean_q_16]
    data_group_3 = [step_goal_size_mean_c_32, step_goal_size_mean_g_32, step_goal_size_mean_q_32]
    data_group_4 = [step_goal_size_mean_c_47, step_goal_size_mean_g_47, step_goal_size_mean_q_47]
    data_group_5 = [step_goal_size_mean_c_53, step_goal_size_mean_g_53, step_goal_size_mean_q_53]
    data_group_6 = [step_goal_size_mean_c_61, step_goal_size_mean_g_61, step_goal_size_mean_q_61]
    data_group_7 = [step_goal_size_mean_c_76, step_goal_size_mean_g_76, step_goal_size_mean_q_76]
    data =  data_group_1 + data_group_2 + data_group_3 + data_group_4 + data_group_5 + data_group_6 + data_group_7    
    plt.boxplot(data, labels=['7-C', '7-G', '7-Q', '16-C', '16-G', '16-Q','32-C', '32-G', '32-Q', '47-C', '47-G', '47-Q', '53-C', '53-G', '53-Q', '61-C', '61-G', '61-Q', 
                             '76-C', '76-G', '76-Q'])
    plt.ylabel('Steps')    
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
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.title('Average shaded plot for all experiments')
    plt.show()
    
    # plottting the rewards versus interquartile range per method
    data = [mean_c, mean_g, mean_q]
    plt.boxplot(data, labels=['Comp RL', 'Goal RL', 'Q learning RL'])
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
    plt.title('Average number of steps per episode')    
    plt.ylabel('Steps')
    plt.xlabel('Episode')
    plt.show()
    
    
    # shift + ctrl + A for multiline comment
    ''' variables = ["Comp RL", "Goal RL", "Q Learning RL"]
    d_1 = [x_7, x_16, x_32, x_47, x_53, x_61, x_76]
    d_2 = [b_7, b_16, b_32, b_47, b_53, b_61, b_76]
    d_1 = [c_7, c_16, c_32, c_47, c_53, c_61, c_76]
    x = ['7', '16', '32', '47', '53', '61', '76']
    
    mean_1 = [np.mean(x_7), np.mean(x_16), np.mean(x_32), np.mean(x_47), np.mean(x_53), np.mean(x_61), np.mean(x_76)]
    mean_2 = [np.mean(b_7), np.mean(b_16), np.mean(b_32), np.mean(b_47), np.mean(b_53), np.mean(b_61), np.mean(b_76)]
    mean_3 = [np.mean(c_7), np.mean(c_16), np.mean(c_32), np.mean(c_47), np.mean(c_53), np.mean(c_61), np.mean(c_76)]
    
    std_1 = [np.std(x_7), np.std(x_16), np.std(x_32), np.std(x_47), np.std(x_53), np.std(x_61), np.std(x_76)]
    std_2 = [np.std(b_7), np.std(b_16), np.std(b_32), np.std(b_47), np.std(b_53), np.std(b_61), np.std(b_76)]
    std_3 = [np.std(c_7), np.std(c_16), np.std(c_32), np.std(c_47), np.std(c_53), np.std(c_61), np.std(c_76)]
    plt.errorbar(x, mean_1, yerr=std_1, fmt='o', capsize=5, label='Comp RL Mean with Std Dev')
    plt.errorbar(x, mean_2, yerr=std_2, fmt='o', capsize=5, label='Goal RL Mean with Std Dev')
    plt.errorbar(x, mean_3, yerr=std_3, fmt='o', capsize=5, label='Q Learning RL Mean with Std Dev')
    #plt.plot(x, mean_1)
    #plt.fill_between(x, mean_1 - std_1, mean_1 + std_1, alpha=0.2)
    #plt.plot(x, mean_2)
    #plt.fill_between(x, mean_2 - std_2, mean_2 + std_2, alpha=0.2)
    #plt.plot(x, mean_3)
    #plt.fill_between(x, mean_3 - std_3, mean_3 + std_3, alpha=0.2)
    plt.ylabel('Rewards')
    plt.xlabel('Goal Sizes')
    plt.title('Mean and the standard deviation of the rewards per goal size')
    plt.show() '''

      



if __name__ == "__main__":
    # tuple_generator()
    #snapshot()
    compare()
    

