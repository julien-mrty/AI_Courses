"""
V) Reinforcement learning and control
Chapter 16
LQR, DDP and LQG
"""


"""
This section introduces finite-horizon Markov Decision Processes (MDPs) and builds on the concepts of value iteration 
and the Bellman equation from previous chapters.

Generalizing MDP Equations :
- In a more general setting, the expectation over future states is used, denoted by E the expectations, instead of summing 
  over discrete states, which works for both discrete and continuous cases.

Finite Horizon MDPs :
- The problem shifts from an infinite horizon (which used a discount factor γ to ensure convergence) to a finite horizon 
  MDP, characterized by a time horizon T. In this setting, rewards are summed over the finite time steps, so no discount 
  factor is needed.
- Unlike infinite-horizon MDPs, the optimal policy in a finite-horizon MDP is generally non-stationary. This means the 
  policy changes over time depending on the remaining time steps, leading to time-dependent transitions and rewards.
  For instance if you have only one step iteration remaining, you will use greedy policy instead of long term 
  expectations.

Time-Dependent Dynamics :
- Finite-horizon MDPs model real-world scenarios better, where things like resource availability and environmental 
  conditions (ex : gas levels, traffic) can change over time. Therefore, both transitions and rewards become 
  time-dependent.
  
Dynamic Programming and Bellman's Equation :
- At the final time step T, the optimal value function is straightforward, computed as the max expected reward of the 
  next action. For earlier time steps, the value function is computed recursively using Bellman's equation.
- The process is solved iteratively, starting from the last time step and working backward to the initial state, 
  ensuring the optimal value function and policy. So instead of choosing an initial state, you chose the final state and
  run the Bellman equations from it.
"""