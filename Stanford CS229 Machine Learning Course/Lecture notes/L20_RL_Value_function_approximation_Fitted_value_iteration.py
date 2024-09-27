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


# Sample n states randomly from the state space S
def sample_states(n_states, state_dim, mean_stddev=0.5):
    #return np.random.randn(n_states, state_dim)
    #return np.random.uniform(0, 1, size=(n_states, state_dim))
    samples = []
    for i in range(n_states):
        # Generate x values between 0 and 1, dividing the interval into n pieces
        x = i / (n_states - 1) # Only using x as the mean of the current sample

        # Generate other values from a normal distribution centered around `x`
        values = []
        for _ in range(state_dim): # -1 because x is also added
            values.append(np.random.normal(loc=x, scale=mean_stddev))

        samples.append(values)

    return np.array(samples)


def reward_function(s):
    if s.ndim > 1:
        return -np.linalg.norm(s, axis=1)

    return -np.linalg.norm(s)


# Transition function to get next states given current state and action
def transition_function(s, a):
    scale = 0.1
    # Apply random perturbations as transitions
    # *s.shape allow to unpack the shape tuple
    noise = np.random.randn(*s.shape) * scale
    return s + a + noise


""" Feature mapping is used to capture non linear relationships with a linear regression model. By adding s^2 it can now
approximate quadratic relationships. As I understand, the bias term (np.ones((s.shape[0], 1))) allows the model to learn 
a constant offset in the value function. Without a bias term, the model would always predict a value that passes through
the origin. You also add a dim to theta, so the model can add a bias so the model do not need to pass at (0, 0). This is 
not useful if the features are normalized. """
def feature_mapping(s):
    return np.concatenate([s, s ** 2, np.ones((s.shape[0], 1))], axis=1) # With current samples, may lead to overfitting
    #return np.concatenate([s, np.ones((s.shape[0], 1))], axis=1)
    #return np.concatenate([s], axis=1)


def gradient_descent(features, y, theta, learning_rate=0.01, num_iters=100):
    n_samples = features.shape[0]
    new_theta = theta.copy()

    for _ in range(num_iters):
        predictions = features @ new_theta
        errors = predictions - y
        gradient = (1 / n_samples) * features.T @ errors
        new_theta -= learning_rate * gradient

    return new_theta


def fitted_value_iteration(n_states, state_dim, actions, n_actions, gamma, max_iter=100, tolerance=1e-2):
    # Sampled states
    states = sample_states(n_states, state_dim)

    # Initialize parameters for linear approximation of value function
    theta_dim = feature_mapping(states).shape[1] # Theta should match the dim of the mapped data
    theta = np.zeros(theta_dim)

    for iteration in range(max_iter):
        y = np.zeros(n_states)  # Target values for supervised learning

        # For each state, compute target values
        for i, s in enumerate(states):
            q_values = np.zeros(n_actions)

            # For each action, sample next states and estimate Q-values
            for a_index, a in enumerate(actions):
                next_states = np.array([transition_function(s, a) for _ in range(10)])  # k next states sampled
                rewards = reward_function(s) # Reward of the current state
                # Estimate V(s_prime) for next states
                next_values = np.dot(feature_mapping(next_states), theta)
                # Estimate Q(a), the average reward we can expect when taking this action
                q_values[a_index] = np.mean(rewards + gamma * next_values)

            # Set y(i) to be the maximum Q-value across actions
            y[i] = np.max(q_values)

        # Linear regression to minimize ||theta^T * phi(s) - y(i)||^2
        features = feature_mapping(states)
        new_theta = gradient_descent(features, y, theta, 0.01)  # Solving linear regression

        # Check for convergence and print progress
        print(f"Iteration {iteration + 1}: Loss = {np.mean(((features @ theta) - y) ** 2)}")

        delta = np.sum(abs(theta - new_theta))
        theta = new_theta

        if delta < tolerance:
            break

    return theta


# Parameters for the MDP
n_states = 100  # Number of sampled states
state_dim = 3  # Dimension of the state space
actions = [-1, 1]
n_actions = 2  # Number of discrete actions
gamma = 0.9  # Discount factor

theta = fitted_value_iteration(n_states, state_dim, actions, n_actions, gamma)
print("Fitted value function parameters:", theta)
