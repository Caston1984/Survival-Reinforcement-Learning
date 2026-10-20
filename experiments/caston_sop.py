import numpy as np
import random
from sympy.logic import SOPform, boolalg
from sympy import Symbol, symbols as Symbols
from collections import defaultdict


### Functions
def sample_random(i, n_goals, bases):
    return np.random.randint(0, 2, n_goals)

def sample_best(i, n_goals, tasks):
    if i<len(tasks):
        return tasks[i]
    else:
        return sample_random(i, n_goals, tasks)

def sample_worst(i, n_goals, tasks):
    if i<n_goals:
        return tasks[i]
    else:
        return sample_random(i, n_goals, tasks)

def get_all(n_goals):
    tasks = []
    for t in range(2**n_goals):
        task = bin(t)[2:]
        task = "".join((["0"]*(n_goals-len(task))))+task
        task = np.array([int(g) for g in list(task)])
        tasks.append(task)
    return tasks

def get_bases(n_goals, regular=True):
    if not regular:
        n_goals+=1 # start from 1
    bases = []
    n=int(np.ceil(np.log2(n_goals)))
    m=(2**n)/2
    for i in range(n):
        bases.append([])
        b=False
        for j in range(0,2**n):
            if j>=n_goals:
                break
            if b:
                bases[i].append(1) #1=True=rmax
            else:
                bases[i].append(0) #0=False=rmin
            if (j+1)%m==0:
                if b:
                    b=False
                else:
                    b=True
        m=m/2
    bases = np.array(bases)
    if not regular:
        bases = bases[:,1:] # start from 1
    return bases

def task_exp(tasks, task): # SOP
    symbols = list(tasks.keys())
    symbols.sort()
    goals = task.keys()
    minterms = []
    for goal in goals:
        if task[goal]:
            minterm = [tasks[j][goal] for j in symbols]
            minterms.append(minterm)              
    exp = SOPform(Symbols(symbols), minterms)
    exp = boolalg.simplify_logic(exp)
    if exp == True:
        exp = Symbols('max')
    if exp == False:
        exp = Symbols('min')
    return exp

def exp_task(tasks, max_task, min_task, exp): 
    task = min_task
    if exp:      
        def convert(exp):
            if type(exp) == Symbol:
                compound = tasks[str(exp)].copy()
            elif type(exp) == boolalg.Or:
                compound = convert(exp.args[0])
                for sub in exp.args[1:]:
                    next = convert(sub)
                    for goal in compound:
                        compound[goal] = max(compound[goal],next[goal])
            elif type(exp) == boolalg.And:
                compound = convert(exp.args[0])
                for sub in exp.args[1:]:
                    next = convert(sub)
                    for goal in compound:
                        compound[goal] = min(compound[goal],next[goal])
            else: # NOT
                compound = convert(exp.args[0])
                for goal in compound:
                    compound[goal] = (max_task[goal] + min_task[goal]) - compound[goal]
            return compound    
        task = convert(exp)
    return task

### Given OR, AND, NOT, you can use this function to compose your skills (in values) according to an expression 
# def exp_value(values, max_wvf, min_wvf, exp):   
#     wvf =  min_wvf
#     if exp:    
#         def convert(exp):
#             if type(exp) == Symbol:
#                 compound = values[str(exp)]
#             elif type(exp) == boolalg.Or:
#                 compound = convert(exp.args[0])
#                 for sub in exp.args[1:]:
#                     compound = OR([compound, convert(sub)])
#             elif type(exp) == boolalg.And:
#                 compound = convert(exp.args[0])
#                 for sub in exp.args[1:]:
#                     compound = AND([compound, convert(sub)])
#             else:
#                 compound = convert(exp.args[0])
#                 compound = NOT(compound, max_wvf, min_wvf)
#             return compound        
#         wvf = convert(exp)
#     return wvf
############################################################################################################################


### Example

symbols = "abcd"
n_goals = len(symbols)
all_tasks = get_all(n_goals)
bases = get_bases(n_goals)
bases = {symbols[task]: {symbols[goal]: bases[task][goal] for goal in range(bases.shape[1])} for task in range(bases.shape[0])} # Convert to dictionary
max_task = {symbols[goal]: 1 for goal in range(n_goals)}
min_task = {symbols[goal]: 0 for goal in range(n_goals)}
task = {symbols[goal]: all_tasks[6][goal] for goal in range(n_goals)} # e.g {'a': 0, 'b': 1, 'c': 1, 'd': 0}
exp = task_exp(bases,task)
task_ = exp_task(bases, max_task, min_task, exp)

print("symbols", symbols)
print("n_goals", n_goals)
print("all_tasks", len(all_tasks), all_tasks)
print("bases", bases)
print("max_task", max_task)
print("min_task", min_task)
print("task", task)
print("exp_from_task", exp)
print("task_from_exp", task_)