# Modelling Survival Through Composition of Reinforcement Learning Policies

[![MSc Dissertation](https://img.shields.io/badge/Degree-MSc%20Computer%20Science-blue)](https://www.wits.ac.za)
[![University](https://img.shields.io/badge/University-Wits%20University-blue)](https://www.wits.ac.za)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org)
[![Framework](https://img.shields.io/badge/RL-Compositional%20%26%20Survival%20RL-orange)](#)

This repository contains the official codebase and experimental evaluation for the Master of Science dissertation **"Modelling Survival Through Composition of Reinforcement Learning Policies"** by **Caston Nyabadza** (School of Computer Science & Applied Mathematics, University of the Witwatersrand, Johannesburg), supervised by **Prof. Benjamin Rosman**, **Dr. Steve James**, and **Dr. Geraud Nangue Tasse**.

---

## 📌 Executive Summary

Traditionally, Reinforcement Learning (RL) focuses on reward maximisation. In **Survival-Oriented Reinforcement Learning (SORL)**, the objective shifts to **maximising operational life expectancy** and remaining within safe homeostatic envelopes while avoiding catastrophic terminal states.

Rather than training survival strategies from scratch, this work introduces the **Hierarchical-Compositional Survival Reinforcement Learning (HCSRL)** framework. HCSRL formulates survival as a **zero-shot composition of pre-trained base skills** via **Extended Value Functions (EVFs)** and **Boolean Task Algebra** ($\wedge, \vee, \neg$).

### Key Contributions
1. **State-Space Bifurcation ($\mathcal{S}^{\mathcal{P}} \times \mathcal{S}^{\mathcal{E}}$):** Decouples internal physiological urgency ($\mathcal{S}^{\mathcal{P}}$, e.g., hydration, nutrition, energy) from external spatial navigation ($\mathcal{S}^{\mathcal{E}}$, e.g., grid coordinates).
2. **Semi-MDP Survival Manager ($\pi_s$):** Operates over an exponentially rich option space ($\Omega = 2^{2^n}$) to arbitrate multi-resource trade-offs across macro-time horizons.
3. **Zero-Shot Low-Level Worker Engine:** Instantiates target option policies ($\pi_c$) zero-shot at the primitive step level without additional training or gradient steps.
4. **Empirical Superiority & Complexity Crossover:** Demonstrates that while flat and goal-oriented RL struggle above 40 environmental goals due to the "Oscillation Trap", HCSRL achieves statistically significant long-term survival persistence ($p < 0.001$).

---

## 🏗️ Repository & Package Architecture

The repository has been refactored into a modular, production-grade Python package structure (`src/` layout):

```text
Survival-Reinforcement-Learning/
├── src/
│   ├── algorithms/              # Core RL & Compositional Libraries
│   │   ├── __init__.py
│   │   ├── q_learning.py        # Flat Standard Q-Learning (Benchmark 1)
│   │   ├── goal_oriented_q.py   # Goal-Oriented Q-Learning (Benchmark 2)
│   │   ├── compositional_q.py  # Boolean Composition Engine via Extended Value Functions
│   │   └── sorl.py             # Survival-Oriented RL Manager & SMDP Logic
│   │
│   ├── environments/            # Environmental Models & State Dynamics
│   │   ├── __init__.py
│   │   ├── gridworld.py         # 20x20 Multi-Resource Gridworld Dynamics
│   │   ├── state_generator.py   # Physiological & Spatial State Generator
│   │   └── states.py            # Discrete State Encoders & Homeostatic Bounds
│   │
│   └── utils/                   # Statistical & Visualisation Utilities
│       ├── __init__.py
│       ├── plotting.py          # Learning Curves & Selection Heatmaps
│       ├── evaluation.py        # Kaplan-Meier Survival Analysis & Log-Rank Tests
│       └── helpers.py           # Core Execution Helpers
│
├── experiments/                 # Reproducible Experiment Entry-Points
│   ├── main_v2.py               # Main Benchmark Suite Execution Entry-Point
│   └── caston_sop.py            # Standard Operating Procedure (SOP) Experiments
│
├── .gitignore                   # Dataset, Checkpoint & Environment Exclusions
├── requirements.txt             # Project Dependencies
└── README.md                    # Project Documentation
