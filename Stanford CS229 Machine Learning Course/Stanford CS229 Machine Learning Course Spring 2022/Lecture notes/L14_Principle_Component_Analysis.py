import numpy as np
from fontTools.afmLib import componentRE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


"""
IV) Unsupervised learning
Chapter 12
Principal component analysis (PCA)
"""


def generate_correlated_data(mean, cov, n_samples):
    # Generate uncorrelated data
    uncorrelated_data = np.random.randn(n_samples, len(mean))

    # Perform Cholesky decomposition on the covariance matrix
    # cov = L * L.T
    L = np.linalg.cholesky(cov)

    # Apply the transformation to obtain correlated data
    # Y = X * L.T
    # This give the correlated data given the cov matrix. Still need to add the desired mean to the data.
    correlated_data = uncorrelated_data @ L.T + mean

    return correlated_data


def plot_3D_data(data):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for the 3D data
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5)

    ax.set_title('3D Data')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')

    plt.show()


def plot_2D_data(data):
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title('2D Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()


def normalize_data(data):
    # Get the dimension of the data
    dim = data.shape[1]

    # First, compute means and variances of the data
    means = np.mean(data, axis=0)
    std_dev = []

    for d in range(dim):
        # Compute variance
        variance = np.mean((data[:, d] - means[d]) ** 2)
        # Compute the standard deviation
        sigma = np.sqrt(variance)
        std_dev.append(sigma)

    # Then, normalize the data
    data_normalized = (data - means) / std_dev

    return data_normalized


def principal_components_analysis(data, k):
    n_samples, dim = data.shape

    # Normalize the data
    data_normalized = normalize_data(data)

    sum_outer_products = np.zeros((dim, dim))

    # Compute the covariance matrix
    for i in range(n_samples):
        outer_product = np.outer(data_normalized[i], data_normalized[i])
        # Compute the sum of outer products
        sum_outer_products += outer_product

    # Compute the average
    empirical_cov_matrix = sum_outer_products / n_samples

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(empirical_cov_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1] # [::-1]: reverses the order of the indices returned by argsort
    # Reordering eigenvectors
    # Each column is an eigenvector corresponding to the eigenvalues
    eigenvectors = eigenvectors[:, sorted_indices]
    # Reordering Eigenvalues
    eigenvalues = eigenvalues[sorted_indices]

    # Select the top k eigenvectors
    top_k_eigenvectors = eigenvectors[:, :k]

    y = []

    # Project the data onto the top k principal components
    for i in range(n_samples):
        y_i = np.matmul(top_k_eigenvectors.T, data_normalized[i])
        y.append(y_i)

    # More straightforward :
    # y = data_normalized @ top_k_eigenvectors

    # For a more readable display
    y = np.array(y)
    print(y)


def main():
    # The reduced dimension of the data
    k = 2

    mean = np.array([2, 1, 3])  # Mean vector
    cov = np.array([[1, 0.8, 0.7],  # Covariance matrix
                    [0.8, 1, 0.9],
                    [0.7, 0.9, 1]])
    n_samples = 100  # Number of samples

    correlated_data = generate_correlated_data(mean, cov, n_samples)

    principal_components_analysis(correlated_data, k)

    #plot_3D_data(correlated_data)


if __name__ == "__main__":
    main()