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


class ContinuousMDPModel:
    def __init__(self, state_dim, action_dim):
        # Initialize A and B matrices randomly for the linear model
        self.A = np.random.randn(state_dim, state_dim)
        self.B = np.random.randn(state_dim, action_dim)
        self.noise_cov = None  # Optional noise covariance matrix for stochastic model

    def fit(self, states, actions, next_states):
        # Stack states and actions for regression, X represent all of our input
        X = np.hstack([states, actions])  # Combine states and actions as input (n_samples, state_dim + action_dim)
        Y = next_states  # Target is the next states (n_samples, state_dim)

        # Perform least-squares regression
        # Solve for A and B s_{t+1} = A * s_t + B * a_t
        theta = np.linalg.lstsq(X, Y, rcond=None)[0]

        # Split the solution back into A and B matrices
        # Theta has a shape of (state_dim + action_dim, state_dim), where the first state_dim rows correspond to the
        # matrix A, which maps the current state s_t to the next state. The remaining action_dim rows correspond to the
        # matrix B, which maps the action a_t to the next state.
        self.A = theta[:states.shape[1], :]  # First 'state_dim' rows correspond to A
        self.B = theta[states.shape[1]:, :]  # Remaining 'action_dim' rows correspond to B

        # We need to transpose B here because B is shape (action_dim, state_dim) but is multiplied by an action with
        # shape (action_dim)
        self.B = self.B.T

    def predict(self, state, action):
        next_state = self.A.dot(state) + self.B.dot(action)

        if self.noise_cov is not None:
            # The noise is centered in 0 and its covariance matrix (self.noise_cov) can be estimated from the data
            next_state += np.random.multivariate_normal(np.zeros(state.shape), self.noise_cov)

        return next_state

    def set_noise(self, noise_cov):
        self.noise_cov = noise_cov


if __name__ == "__main__":
    # Parameters
    state_dim = 4  # E.g., for a system with 4 state variables (like x, y, velocity)
    action_dim = 2  # E.g., for a system with 2 action variables (like force or torque)

    # Simulate some example data for training
    n_samples = 10
    states = np.random.randn(n_samples, state_dim)  # Random states s_t
    actions = np.random.randn(n_samples, action_dim)  # Random actions a_t
    next_states = np.random.randn(n_samples, state_dim)  # Random next states s_{t+1}

    # Initialize the MDP model
    mdp_model = ContinuousMDPModel(state_dim, action_dim)

    # Train the model
    mdp_model.fit(states, actions, next_states)

    # Predict the next state for a given state and action
    test_state = np.random.randn(state_dim)
    test_action = np.random.randn(action_dim)
    predicted_next_state = mdp_model.predict(test_state, test_action)

    print("Predicted next state : ", predicted_next_state)

    # Predict the next state
    stochastic_next_state = mdp_model.predict(test_state, test_action)
    print("Stochastic next state : ", stochastic_next_state)

    # Set noise for stochastic predictions
    noise_cov = np.eye(state_dim) * 0.1  # Small Gaussian noise
    # With noise the results are poor
    mdp_model.set_noise(noise_cov)

    # Predict the next state with noise
    stochastic_next_state = mdp_model.predict(test_state, test_action)
    print("Stochastic next state with noise : ", stochastic_next_state)
