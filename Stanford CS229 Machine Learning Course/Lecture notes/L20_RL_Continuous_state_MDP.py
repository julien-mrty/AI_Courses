import numpy as np


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


# Define MDP parameters
n_states = 3
n_actions = 2
gamma = 0.9
threshold = 1e-1

# Transition probabilities P[s, a, s']
P = np.array([
    [[0.8, 0.2, 0.0], [0.0, 1.0, 0.0]],  # From state 0
    [[1.0, 0.0, 0.0], [0.7, 0.0, 0.3]],  # From state 1
    [[0.1, 0.0, 0.9], [1.0, 0.0, 0.0]]   # From state 2
])

print("P real : \n", P)

# Rewards R[s, a, s']
R = np.array([
    [[1, 2, 0], [0, 3, 0]],  # From state 0
    [[3, 0, 1], [0, 0, 2]],  # From state 1
    [[2, 0, 3], [0, 1, 0]]   # From state 2
])


def arrival_state(s, action):
    # Vector listing the states
    states = list(range(n_states))

    # Define the probabilities for each state (must sum to 1)
    probabilities = P[s][action]

    # Randomly select a state based on the probabilities
    s_prime = np.random.choice(states, p=probabilities)

    return s_prime


def compute_state_transition_probabilities(policy, states_reached_count, actions_from_states_count, iterations=100):
    for _ in range(iterations):
        for s in range(n_states):
            # Get the action to take from the policy
            action = policy[s]
            # Sum up the actions take from this state
            actions_from_states_count[s][action] += 1
            # Get s_prime based on the real probabilities
            s_prime = arrival_state(s, action)
            # Update the number of this s_prime have been reached from state s with action a
            states_reached_count[s][action][s_prime] += 1

    return states_reached_count, actions_from_states_count


def value_iteration(V, P, R, gamma, threshold):
    iteration = 0

    while True:
        delta = 0

        for s in range(n_states):
            action_values = np.zeros(n_actions)

            for a in range(n_actions):
                action_values[a] = np.sum(P[s, a] * (R[s, a] + gamma * V))

            new_V = np.max(action_values)
            delta = max(delta, np.abs(new_V - V[s]))
            V[s] = new_V

        iteration += 1
        if delta < threshold:
            break

    return V


def update_policy(V, P, R, gamma):
    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):
        action_values = np.zeros(n_actions)

        for a in range(n_actions):
            action_values[a] = np.sum(P[s, a] * (R[s, a] + gamma * V))

        policy[s] = np.argmax(action_values)

    return policy


def policy_evaluation(policy, P, R, gamma, threshold=1e-3):
    V = np.zeros(n_states)

    while True:
        delta = 0

        for s in range(n_states):
            v = V[s]
            a = policy[s]
            V[s] = np.sum(P[s, a] * (R[s, a] + gamma * V))
            delta = max(delta, np.abs(v - V[s]))

        if delta < threshold:
            break

    return V


def estimate_PSA(states_reached_count, actions_from_states_count):
    # Initialize policy randomly
    random_policy = np.random.randint(n_actions, size=n_states)
    # Initialize P_estimate with zeros
    P_estimate = np.zeros((n_states, n_actions, n_states))

    # Get the total number of states reached and actions taken
    states_reached_count, actions_from_states_count = compute_state_transition_probabilities(random_policy, states_reached_count, actions_from_states_count)

    # Normalize to get probabilities
    for s in range(n_states):
        for a in range(n_actions):
            if actions_from_states_count[s][a] > 0:
                P_estimate[s][a] = states_reached_count[s][a] / actions_from_states_count[s][a]

    return P_estimate


def model_learning(max_iter=100):
    # Initialize value function
    V = np.zeros(n_states)
    # Initialize the computed transitions probabilities
    P_estimate = np.zeros((n_states, n_actions, n_states))
    # Initialize the policy randomly
    random_policy = np.random.randint(n_actions, size=n_states)

    # Counter of the number of times each state has been reached
    states_reached_count = np.zeros((n_states, n_actions, n_states))
    # Counter of the number of times each action has been taken from each state
    actions_from_states_count = np.zeros((n_states, n_actions))

    policy_eval = policy_evaluation(random_policy, P_estimate, R, gamma)

    for iteration in range(max_iter):
        # Compute the PSA
        new_P_estimate = estimate_PSA(states_reached_count, actions_from_states_count)

        # Compute optimal value function
        V = value_iteration(V, new_P_estimate, R, gamma, threshold)
        
        # Compute the policy
        new_policy = update_policy(V, new_P_estimate, R, gamma)

        # Compute the new policy evaluation
        new_policy_eval = policy_evaluation(new_policy, new_P_estimate, R, gamma)
        # Get the difference with the old evaluation
        delta = np.sum(np.abs(new_policy_eval - policy_eval))

        # Update with the computed values for this step
        P_estimate = new_P_estimate
        policy = new_policy

        if delta < threshold:
            print(f"Model learning converged in : {iteration} iterations." )
            break

    return P_estimate, V, policy


P, V, policy = model_learning()
print("===================== Results")
print("P_estimate : \n", P)
print("V : \n", V)
print("policy : \n", policy)