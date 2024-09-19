"""
V) Reinforcement learning and control
Chapter 15
Reinforcement learning
"""


"""
Markov Decision Processes (MDP)

- S : The set of states
- A : The set of actions
- P_sa : The state transition probabilities
- gamma (ga) : The discount factor
- R : The reward function

MDP proceeds as follow : 
Start in state s0, chose some actions a0 -> MDP randomly transition to some successor state s1 according to s1 ~ P_s0a0.
Then pick another action and so on.

Can be represented as follow :  
    a0       a1       a2       a3
s0 ----> s1 ----> s2 ----> s3 ----> ...

Our total payoff is given by : 
R(s0, a0) + ga * R(s1, a1) + (ga^2) * R(s2, a2) + ...
Simpler version : 
R(s0) + ga * R(s1) + (ga^2) * R(s2) + ...

The goal in RL is to choose actions to maximise the expected value of total payoff :
E[ R(s0) + ga * R(s1) + (ga^2) * R(s2) + ... ]

A policy is a function pi : S -> A mapping the states to the actions.

The value function of a policy pi is :
V^pi_(s) = E[ R(s0) + ga * R(s1) + (ga^2) * R(s2) + ... | s0 = s, pi ]

Check main notes page 178 to get Bellman equations. It is an equation to compute the value function.
It is used to efficiently solve V^pi. Specifically in a finite state MDP, we can write down one such equation for the 
V^pi(s) for every state s.

The optimal value function is defined as : 
V^*(s) = max_pi V^pi(s)
This is the best possible expected sum of rewards using any policy. 
Check main notes page 179 to get the Bellman equations for the optimal value function. 

There is also another equation that gives us the best policy pi^* also noted pi^(s)
Check main notes page 179 equation 15.3
"""