import numpy as np
from fontTools.afmLib import componentRE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


"""
IV) Unsupervised learning
Chapter 13
Independent components analysis (PCA)
"""

"""
As a motivating example, consider the \cocktail party problem." Here, d
speakers are speaking simultaneously at a party, and any microphone placed
in the room records only an overlapping combination of the d speakers' voices.
"""

"""
To understand what we have and what we are looking for :
- s(i) is an d-dimensional vector, and s(i)_j is the sound that speaker j was uttering at time i. 
- x(i) in an  d-dimensional vector, and x(i) j is the acoustic reading recorded by microphone j at time i.
"""


#np.random.seed(4)  # For reproducibility


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def compute_log_likelihood(data, w):
    # Compute the sources (S = W*X)
    S = np.dot(w, data.T)

    # Compute the log-likelihood
    log_density = np.sum(np.log(sigmoid_prime(S) + 1e-10), axis=0)  # sum over components
    log_likelihood = np.mean(log_density)

    # Add log determinant term
    log_det_w = np.log(np.abs(np.linalg.det(w)) + 1e-10) # Add small values to ensure calculus stability
    log_likelihood += log_det_w

    return log_likelihood


def batch_gradient_ascent(learning_rate, w, data):
    n_samples, dim = data.shape

    # Calculate the sources (S = W*X)
    S = np.dot(w, data.T)

    # Compute the gradient for each component
    sigmoid_grad = 1 - 2 * sigmoid(S)
    gradient = np.dot(sigmoid_grad, data) / n_samples  # averaged over samples

    # Update w using gradient ascent
    w += learning_rate * (gradient + np.linalg.inv(w))

    return w


def one_sample_gradient_ascent(learning_rate, w, data):
    n_samples, dim = data.shape

    # Compute the gradient and update w sample by sample
    # Better result but less optimized
    for i in range(n_samples):
        # Gradient from the log-likelihood derivative
        gradient = (1 - 2 * sigmoid(np.dot(w, data[i, :]))) @ data[i].T

        # Update w using gradient ascent
        w += learning_rate * (gradient + np.linalg.inv(w))

    return w


def independent_components_analysis(learning_rate, data, sources, tolerance=1e-3, max_iter=1000):
    n_speakers = sources.shape[1]
    n_microphones = data.shape[1]

    # Initialize W (square matrix for easier calculus)
    w = np.random.uniform(0, 1, (n_speakers, n_microphones))
    #w = np.random.rand(n_speakers, n_microphones) * 0.01

    log_likelihood = []

    for iteration in range(max_iter):
        # Update w values trough gradient ascent
        w = one_sample_gradient_ascent(learning_rate, w, data)

        # Compute the log likelihood
        log_likelihood.append(compute_log_likelihood(data, w))

        # Check for convergence
        if iteration > 0 and abs(log_likelihood[-1] - log_likelihood[-2]) < tolerance:
            print(
                f"============================================ Converged in {iteration + 1} iterations, \nlog_likelihood = {log_likelihood[-1]}, \nW = {w}")
            break

        print(
            f"============================================ Iterations {iteration + 1}, \nlog_likelihood = {log_likelihood[-1]}, \nW = {w}")

    return w, log_likelihood

def main():
    n_samples = 10 # number of samples in the record
    # Same number of microphones and speakers for simplicity of the calculus (loglikelihood)
    n_microphones = n_speakers = 3 # number of microphones and speakers
    low = 0
    high = 1

    learning_rate = 0.01

    sources = np.zeros((n_samples, n_speakers)) # The source that generated our data
    data = np.random.uniform(low, high, (n_samples, n_microphones)) # x the data
    print("Sources : \n", sources)
    print("Data : \n", data)

    # For one sample (i), the source (j) equals to : s(i)(j) = w(j) * x(i)

    independent_components_analysis(learning_rate, data, sources)


if __name__ == "__main__":
    main()
