import numpy as np
import random
from sympy.logic import SOPform, boolalg
from sympy import Symbol, symbols as Symbols
from collections import defaultdict
import numpy as np
from copy import deepcopy
from ast import literal_eval


class Tasks:

    def __init__(self, symbols):
        self.symbols = symbols
        self.n_goals = len(self.symbols)
        self.tasks = []
        self.tasks_dic = {}
        self.bases = []
        self.bases_dic = {}
        #self.tasks = self.get_all(self.n_goals)
        #self.base = {self.symbols[task]: {symbols[goal]: self.get_bases(self.n_goals)[task][goal] for goal in range(self.get_bases(self.n_goals).shape[1])} for task in
        #             range(self.get_bases(self.n_goals).shape[0])}  # Convert to dictionary

    ### Functions
    def sample_random(self):
        return np.random.randint(0, 2, self.n_goals)

    def sample_best(self, i):
        if i < len(self.tasks):
            return self.tasks[i]
        else:
            return self.sample_random(i, self.n_goals, self.tasks)

    def sample_worst(self, i):
        if i < self.n_goals:
            return self.tasks[i]
        else:
            return self.sample_random(i, self.n_goals, self.tasks)

    def get_all(self):
        #tasks = [] made it a class variable
        for t in range(2 ** self.n_goals):
            self.task = bin(t)[2:]
            self.task = "".join((["0"] * (self.n_goals - len(self.task)))) + self.task
            self.task = np.array([int(g) for g in list(self.task)])
            self.tasks.append(self.task)
        return self.tasks

    def task_dic(self):
        for key in range(len(self.tasks)):
            self.tasks_dic[key] = {}
            for goal, i in zip(self.symbols, range(len(self.symbols))):
                self.tasks_dic[key][goal] = self.tasks[key][i]
        return self.tasks_dic

    def get_bases(self, regular=True):
        #This allows us to formulate new tasks in terms of the negation, disjunction and conjunction of a set of base tasks.
        if not regular:
            self.n_goals += 1  # start from 1
        #bases = [] because I have made it a class variable
        n = int(np.ceil(np.log2(self.n_goals)))
        m = (2 ** n) / 2
        for i in range(n):
            self.bases.append([])
            b = False
            for j in range(0, 2 ** n):
                if j >= self.n_goals:
                    break
                if b:
                    self.bases[i].append(1)  # 1=True=rmax
                else:
                    self.bases[i].append(0)  # 0=False=rmin
                if (j + 1) % m == 0:
                    if b:
                        b = False
                    else:
                        b = True
            m = m / 2
        self.bases = np.array(self.bases)
        if not regular:
            self.bases = self.bases[:, 1:]  # start from 1
        return self.bases

    def base_dic(self):
        self.bases_dic =  {self.symbols[task]: {self.symbols[goal]: self.bases[task][goal] for goal in range(self.bases.shape[1])} for task in
         range(self.bases.shape[0])}  # Convert to dictionary
        return self.bases_dic

    def task_exp(self, task):  # SOP
        symbols = list(self.base_dic().keys())
        symbols.sort()
        goals = task.keys()
        minterms = []
        for goal in goals:
            if task[goal]:
                minterm = [self.bases_dic[j][goal] for j in symbols]
                minterms.append(minterm)
        exp = SOPform(Symbols(symbols), minterms)
        exp = boolalg.simplify_logic(exp)
        if exp == True:
            exp = Symbols('max')
        if exp == False:
            exp = Symbols('min')
        return exp

    def exp_task(self, tasks, max_task, min_task, exp):
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
                            compound[goal] = max(compound[goal], next[goal])
                elif type(exp) == boolalg.And:
                    compound = convert(exp.args[0])
                    for sub in exp.args[1:]:
                        next = convert(sub)
                        for goal in compound:
                            compound[goal] = min(compound[goal], next[goal])
                else:  # NOT
                    compound = convert(exp.args[0])
                    for goal in compound:
                        compound[goal] = (max_task[goal] + min_task[goal]) - compound[goal]
                return compound

            task = convert(exp)
        return task


class Composition:
    def __init__(self, values, max_wvf, min_wvf, exp):
        self.values = values
        self.max_wvf = max_wvf
        self.min_wvf = min_wvf
        self.exp = exp

    #########################################################################################
    def Q_equal(self, Q1, Q2, epsilon=1e-5):
        for state in Q1:
            for action in range(len(Q1[state])):
                v1 = Q1[state][action]
                v2 = Q2[state][action]
                if abs(v1 - v2) > epsilon:
                    return False
        return True

    def EQ_equal(self, EQ1, EQ2, epsilon=1e-5):
        for state in EQ1:
            for goal in EQ1[state]:
                for action in range(len(EQ1[state][goal])):
                    v1 = EQ1[state][goal][action]
                    v2 = EQ2[state][goal][action]
                    if not (abs(v1 - v2) < epsilon or (v1 < -30 and v2 < -30)):
                        return False
        return True

    #########################################################################################
    def epsilon_greedy_policy_improvement(env, Q, epsilon=1):
        """
        Implements policy improvement by acting epsilon-greedily on Q
        Arguments:
        env -- environment with which agent interacts
        Q -- Action function for current policy
        epsilon -- probability
        Returns:
        policy_improved -- Improved policy
        """

        def policy_improved(state, epsilon=epsilon):
            probs = np.ones(env.action_space.n, dtype=float) * (epsilon / env.action_space.n)
            best_action = np.random.choice(np.flatnonzero(Q[state] == Q[state].max()))  # np.argmax(Q[state]) #
            probs[best_action] += 1.0 - epsilon
            return probs

        return policy_improved

    def epsilon_greedy_generalised_policy_improvement(self, env, Q, epsilon=1):
        """
        Implements generalised policy improvement by acting epsilon-greedily on Q
        Arguments:
        env -- environment with which agent interacts
        Q -- Action function for current policy
        Returns:
        policy_improved -- Improved policy
        """

        def policy_improved(state, goal=None, epsilon=epsilon):
            probs = np.ones(env.action_space.n, dtype=float) * (epsilon / env.action_space.n)
            values = [Q[state][goal]] if goal else [Q[state][goal] for goal in Q[state].keys()]
            if len(values) == 0:
                best_action = np.random.randint(env.action_space.n)
            else:
                values = np.max(values, axis=0)
                best_action = np.random.choice(np.flatnonzero(values == values.max()))
            probs[best_action] += 1.0 - epsilon
            return probs

        return policy_improved

    #########################################################################################
    def Q_learning(self, env, Q_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100, maxstep=100):
        """
        Implements Q_learning
        Arguments:
        env -- environment with which agent interacts
        gamma -- discount factor
        alpha -- learning rate
        maxiter -- maximum number of episodes
        Returns:
        Q -- New estimate of Q function
        """
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        behaviour_policy = epsilon_greedy_policy_improvement(env, Q, epsilon=epsilon)

        stop_cond = lambda k: k < maxiter
        if Q_optimal:
            stop_cond = lambda _: not Q_equal(Q_optimal, Q)

        stats = {"R": [], "T": 0}
        k = 0
        T = 0
        state = env.reset()
        stats["R"].append(0)
        while stop_cond(k):
            probs = behaviour_policy(state, epsilon=epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            state_, reward, done, _ = env.step(action)

            stats["R"][k] += reward

            G = 0 if done else np.max(Q[state_])
            TD_target = reward + gamma * G
            TD_error = TD_target - Q[state][action]
            Q[state][action] = Q[state][action] + alpha * TD_error

            state = state_
            T += 1
            if done:
                state = env.reset()
                stats["R"].append(0)
                k += 1
        stats["T"] = T

        return Q, stats

    def Goal_Oriented_Q_learning(self, env, T_states=None, Q_optimal=None, gamma=1, epsilon=1, alpha=1, maxiter=100,
                                 maxstep=100):
        """
        Implements Goal Oriented Q_learning
        Arguments:
        env -- environment with which agent interacts
        gamma -- discount factor
        alpha -- learning rate
        maxiter -- maximum number of episodes
        Returns:
        Q -- New estimate of Q function
        """
        N = min(env.rmin, (env.rmin - env.rmax) * env.diameter)
        Q = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
        behaviour_policy = self.epsilon_greedy_generalised_policy_improvement(env, Q, epsilon=epsilon)

        sMem = {}  # Goals memory
        if T_states:
            for state in T_states:
                sMem[str(state)] = 0

        stop_cond = lambda k: k < maxiter
        if Q_optimal:
            stop_cond = lambda _: not self.EQ_equal(Q_optimal, Q)

        stats = {"R": [], "T": 0}
        k = 0
        T = 0
        state = env.reset()
        stats["R"].append(0)
        while stop_cond(k):
            probs = behaviour_policy(state, epsilon=epsilon)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            state_, reward, done, _ = env.step(action)

            stats["R"][k] += reward

            if done:
                sMem[state] = 0

            for goal in sMem.keys():
                if state != goal and done:
                    reward_ = N
                else:
                    reward_ = reward

                G = 0 if done else np.max(Q[state_][goal])
                TD_target = reward_ + gamma * G
                TD_error = TD_target - Q[state][goal][action]
                Q[state][goal][action] = Q[state][goal][action] + alpha * TD_error

            state = state_
            T += 1
            if done:
                state = env.reset()
                stats["R"].append(0)
                k += 1
        stats["T"] = T

        return Q, stats

    #########################################################################################

    #########################################################################################
    def EQ_NP(self, EQ):
        P = defaultdict(lambda: defaultdict(lambda: 0))
        for state in EQ:
            for goal in EQ[state]:
                P[state][goal] = np.argmax(EQ[state][goal])
                # v = EQ[state][goal]
                # P[state][goal] = np.random.choice(np.flatnonzero(v == v.max()))
        return P

    def EQ_P(self, EQ, goal=None):
        P = defaultdict(lambda: 0)
        for state in EQ:
            if goal:
                P[state] = np.argmax(EQ[state][goal])
                # v = EQ[state][goal]
                # P[state] = np.random.choice(np.flatnonzero(v == v.max()))
            else:
                Vs = [EQ[state][goal] for goal in EQ[state].keys()]
                P[state] = np.argmax(np.max(Vs, axis=0))
                # v = np.max(Vs,axis=0)
                # P[state] = np.random.choice(np.flatnonzero(v == v.max()))
        return P

    def Q_P(self, Q):
        P = defaultdict(lambda: 0)
        for state in Q:
            P[state] = np.argmax(Q[state])
        return P

    def EQ_NV(self,EQ):
        V = defaultdict(lambda: defaultdict(lambda: 0))
        for state in EQ:
            for goal in EQ[state]:
                V[state][goal] = np.max(EQ[state][goal])
        return V

    def EQ_V(self, EQ, goal=None):
        V = defaultdict(lambda: 0)
        for state in EQ:
            if goal:
                V[state] = np.max(EQ[state][goal])
            else:
                Vs = [EQ[state][goal] for goal in EQ[state].keys()]
                V[state] = np.max(np.max(Vs, axis=0))
        return V

    def NV_V(self, NV, goal=None):
        V = defaultdict(lambda: 0)
        for state in NV:
            if goal:
                V[state] = NV[state][goal]
            else:
                Vs = [NV[state][goal] for goal in NV[state].keys()]
                V[state] = np.max(Vs)
        return V

    def Q_V(self, Q):
        V = defaultdict(lambda: 0)
        for state in Q:
            V[state] = np.max(Q[state])
        return V

    def EQ_Q(self, EQ, goal=None):
        Q = defaultdict(lambda: np.zeros(5))
        for state in EQ:
            if goal:
                Q[state] = EQ[state][goal]
            else:
                Vs = [EQ[state][goal] for goal in EQ[state].keys()]
                Q[state] = np.max(Vs, axis=0)
        return Q

    #########################################################################################
    def MAX(self, Q1, Q2):
        Q = defaultdict(lambda: 0)
        for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
            Q[s] = np.max([Q1[s], Q2[s]], axis=0)
        return Q

    def AVG(self, Q1, Q2):
        Q = defaultdict(lambda: 0)
        for s in list(set(list(Q1.keys())) & set(list(Q2.keys()))):
            Q[s] = (Q1[s] + Q2[s]) / 2
        return Q


    #########################################################################################
    def EQMAX(self, EQ, rmax=2):  # Estimating EQ_max
        rmax = rmax
        EQ_max = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
        for s in list(EQ.keys()):
            for g in list(EQ[s].keys()):
                c = rmax - max(EQ[g][g])
                if s == g:
                    EQ_max[s][g] = EQ[s][g] * 0 + rmax
                else:
                    EQ_max[s][g] = EQ[s][g] + c
        return EQ_max

    def EQMIN(self, EQ, rmin=-0.1):  # Estimating EQ_min
        rmin = rmin
        EQ_min = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
        for s in list(EQ.keys()):
            for g in list(EQ[s].keys()):
                c = rmin - max(EQ[g][g])
                if s == g:
                    EQ_min[s][g] = EQ[s][g] * 0 + rmin
                else:
                    EQ_min[s][g] = EQ[s][g] + c
        return EQ_min
    def NOT(self, EQ, EQ_max=None, EQ_min=None):
        EQ_max = EQ_max if EQ_max else self.EQMAX(EQ)
        EQ_min = EQ_min if EQ_min else self.EQMIN(EQ)
        EQ_not = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
        for s in list(EQ.keys()):
            for g in list(EQ[s].keys()):
                EQ_not[s][g] = (EQ_max[s][g] + EQ_min[s][g]) - EQ[s][g]
        return EQ_not

    def OR(self, EQ1, EQ2):
        EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
        for s in list(EQ1.keys()):
            for g in list(EQ1[s].keys()):
                EQ[s][g] = np.max([EQ1[s][g], EQ2[s][g]], axis=0)
        return EQ

    def AND(self, EQ1, EQ2):
        EQ = defaultdict(lambda: defaultdict(lambda: np.zeros(5)))
        for s in list(EQ1.keys()):
            for g in list(EQ1[s].keys()):
                EQ[s][g] = np.min([EQ1[s][g], EQ2[s][g]], axis=0)
        return EQ

    def exp_task(self, tasks, max_task, min_task, exp):
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
                            compound[goal] = max(compound[goal], next[goal])
                elif type(exp) == boolalg.And:
                    compound = convert(exp.args[0])
                    for sub in exp.args[1:]:
                        next = convert(sub)
                        for goal in compound:
                            compound[goal] = min(compound[goal], next[goal])
                else:  # NOT
                    compound = convert(exp.args[0])
                    for goal in compound:
                        compound[goal] = (max_task[goal] + min_task[goal]) - compound[goal]
                return compound

            task = convert(exp)
        return task

    ### Given OR, AND, NOT, you can use this function to compose your skills (in values) according to an expression
    def exp_value(self, values, max_wvf, min_wvf, exp):
         wvf =  min_wvf
         if exp:
             def convert(exp):
                 if type(exp) == Symbol:
                     compound = values[str(exp)]
                 elif type(exp) == boolalg.Or:
                     compound = convert(exp.args[0])
                     for sub in exp.args[1:]:
                         compound = self.OR([compound, convert(sub)])
                 elif type(exp) == boolalg.And:
                     compound = convert(exp.args[0])
                     for sub in exp.args[1:]:
                         compound = self.AND([compound, convert(sub)])
                 else:
                     compound = convert(exp.args[0])
                     compound = self.NOT(compound, max_wvf, min_wvf)
                 return compound
             wvf = convert(exp)
         return wvf
    ############################################################################################################################


