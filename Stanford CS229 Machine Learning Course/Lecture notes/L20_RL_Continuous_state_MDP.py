"""
V) Reinforcement learning and control
Chapter 15
Reinforcement learning
"""

"""
Lecture 20 is the direct continuous of lecture 17. 
Lecture 18 is "Societal impact of ML". 
There is no lecture 19 on Stanford youtube channel for 2022 lectures.
"""


"""
Continuous state MDPs
"""


"""
Discretization 

With infinite state spaces, particularly in cases where the state space is continuous (ex : a car’s position, 
orientation, and velocity). One approach to solving continuous-state MDPs is discretization, where the continuous state 
space is divided into discrete cells, and methods like value iteration or policy iteration are applied to the 
discretized MDP.

However, discretization has limitations :
- Naive Value Function Representation : It assumes the value function is constant within each grid cell, which is 
  unsuitable for smooth functions. This leads to poor approximations unless the grid is very fine (grid granularity
  issue).
- Curse of Dimensionality : As the dimensionality of the state space increases, the number of discrete states grows 
  exponentially. For example, discretizing a 10-dimensional state space with 100 values per dimension results in 10^20 
  states, which is computationally infeasible.

Discretization works well for 1D or 2D problems and can be extended up to 4D with careful planning, but becomes 
impractical for higher dimensions.
"""