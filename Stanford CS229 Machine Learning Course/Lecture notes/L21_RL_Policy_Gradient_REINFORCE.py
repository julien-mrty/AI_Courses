"""
V) Reinforcement learning and control
Chapter 17
Policy Gradient (REINFORCE)
"""


"""
Policy gradient method.

Key concepts :
- In reinforcement learning, the goal is to find a policy that maximizes the cumulative reward.
- A policy pi is a probability distribution over actions a_t given the current state s_t, parameterized by θ.
- The policy gradient method updates θ in the direction of the gradient of the expected return J(θ).


REINFORCE algorithm : 
- REINFORCE is a model-free algorithm (don't need to know its environment, learn from reward it gets after each action) 
  for policy optimization in reinforcement learning.
- It directly optimizes a randomized policy.
- The focus is on a finite horizon setting.

- Learning without transition probability :
    - It only assumes access to samples from transition probabilities and queries to the reward function, without requiring 
      explicit knowledge of their analytical forms
    - The algorithm learns the policy π_θ(a∣s), representing the probability of taking action a in state s based on 
      parameter θ.
      
- Objective function :
    - The goal is to optimize the expected total reward over a trajectory J(θ).
    
- Policy Gradient Estimation:
    - To optimize J(θ) using gradient ascent, the algorithm needs to compute ∇J(θ) without knowing the form of the 
      reward functon or transition probabilities.
    - The gradient is estimated using the likelihood-ratio trick. Check the lecture notes from page 208.

- The core of the REINFORCE algorithm is to update policy parameters based on sampled trajectories ∇_θ J(θ)
- Intuitively, this means adjusting the policy to favor actions leading to high rewards while downplaying those leading 
  to low rewards.
  

- The policy gradient formula (17.8, p209) :
    - LHS : Represents how much the policy is to changes in the parameter θ for the action taken a_t given the state 
      s_t. It shows how the probability of selecting action a_t would change when adjusting θ.
    - RHS : Represents the total reward collected over the trajectory. It acts as a weight that indicates how rewarding 
      this trajectory (or sequence of actions) was. If a trajectory has a high reward, we want to adjust the policy in 
      the direction that makes it more likely to repeat this trajectory. Conversely, if a trajectory has a low reward, 
      we don't want to emphasize actions that led to it as much.

  
- Equation (17.9, p210) :
RHS = 1 refer to lecture's example
E[LHS * RHS] = 0, if RHS = 1, or the policy is already maximized.
This means that their is no more updates to make on order to optimized the policy. 
As I understand, if the policy is already maximized, the derivative of the policy for state a is equal to 0 (top of the
curve, already the most probable action to take, the gradient of the policy with respect to θ will be 0 at the optimal 
point). 
If the reward function is a constant (=1) at all states, the mean (the expectation) of the update on theta is equal to 0 
because no matter which action you take, you will end up on the state with a reward function equal to 1. There is no 
improvement to be made.


- Equation (17.10, p210) :


Baseline to Reduce Variance :
- A significant challenge with REINFORCE is the high variance of gradient estimates.
- To reduce this variance, the method introduces a baseline B(s_t), which does not affect the gradient’s expectation but
  lowers its variance.
- A good choice for the baseline is the value function V^π(s_t), which represents the expected future reward from state 
  s_t.
- It captures the idea of comparing the expected future rewards to a baseline B(s_t) "baseline reward". Knowing that 
  B(s_t) si already the expected future rewards, it compares expected future reward B(s_t) to current future reward
  of the current policy. This means that if the future rewards exceed the baseline, it indicates that the current policy
  is performing better than expected. Conversely, if the sum is less than the baseline, it suggests that the policy 
  needs adjustment to achieve better performance.


Algorithmic Steps:
- Vanilla REINFORCE involves the following iterative steps :
    Loop until convergence.
    1) Collect Trajectories : Sample trajectories by executing the current policy.
    2) Estimate Baseline : Fit a baseline function B(s_t) to minimize the squared error between the estimated future 
    rewards and the actual rewards.
    3) Policy Update : Use the gradient estimate to update the policy parameters via gradient ascent.


Summary:
- REINFORCE relies on empirical sampling of trajectories and does not require explicit knowledge of the environment 
  dynamics.
- The gradient step adjusts the policy to increase the likelihood of high-reward trajectories and decrease the 
  likelihood of low-reward trajectories.
- The algorithm has a high variance in its gradient estimates, but the use of a baseline mitigates this to some extent.
- While REINFORCE is conceptually simple, its slow convergence and high variance can be drawbacks in practice.
"""