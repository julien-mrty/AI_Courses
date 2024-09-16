import numpy as np


"""
IV) Unsupervised learning
Chapter 10
Expectation Maximisation algorithm
"""


"""
Contrary to k-means algo, the EM algo make soft guesses for the values of z(i) (the probability that each example 
belongs to one of the k gaussian density function). Each example have to probability to be each gaussian PDF, instead of
belonging to one of the gaussian PDF (like in the k-means).
"""


def generate_mixture_of_gaussians(n_samples, num_cluster, phi, means, covariances):
    d = means.shape[1]  # Dimensionality of the data
    X = np.zeros((n_samples, d))  # Data points
    z = np.zeros(n_samples, dtype=int)  # Latent variables (cluster assignments)

    # Step 1: Sample latent variables z(i) from a Multinomial(phi)
    for i in range(n_samples):
        z[i] = np.random.choice(num_cluster, p=phi)  # Choose which Gaussian component to sample from

        # Step 2: Sample x(i) from the Gaussian N(mean_j, cov_j) where j = z(i)
        X[i, :] = np.random.multivariate_normal(means[z[i]], covariances[z[i]])

    return X, z


# Compute the 2D Gaussian probability density function (PDF)
def multivariate_gaussian_pdf(x, mean, cov):
    d = mean.shape[0]  # Dimensionality
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)

    norm_factor = 1.0 / (np.sqrt((2 * np.pi) ** d * cov_det))

    diff = x - mean
    exponent = -0.5 * np.sum(np.dot(diff, cov_inv) * diff, axis=1)

    return norm_factor * np.exp(exponent)


# Compute the probability w, that each example belongs to each cluster
def e_step(num_clusters, X, phi, mean, cov):
    n_samples = X.shape[0]

    # Initialize the responsibility matrix (w) with zeros
    w = np.zeros((n_samples, num_clusters))

    # Calculate the numerator for each Gaussian component
    for j in range(num_clusters):
        w[:, j] = phi[j] * multivariate_gaussian_pdf(X, mean[j], cov[j])

    # Normalize the responsibilities across clusters for each data point
    w /= np.sum(w, axis=1, keepdims=True)

    return w


def m_step(num_clusters, X, w):
    n_samples, n_features = X.shape

    # Initialize updated parameters
    phi = np.zeros(num_clusters)
    mean = np.zeros((num_clusters, n_features))
    cov = [np.zeros((n_features, n_features)) for _ in range(num_clusters)]

    for j in range(num_clusters):
        # Update the mixture of weights
        phi[j] = np.sum(w[:, j]) / n_samples

        # Update the means
        weighted_sum = np.sum(w[:, j, np.newaxis] * X, axis=0)  # np.newaxis adds a new dimension for broadcasting
        mean[j] = weighted_sum / np.sum(w[:, j])

        # Update the covariances
        diff = X - mean[j]
        weighted_diff = w[:, j][:, np.newaxis] * diff
        cov[j] = np.dot(weighted_diff.T, diff) / np.sum(w[:, j])

    return phi, mean, cov


def em_algo(X, num_clusters, tolerance, max_iters=100):
    n_samples, d = X.shape  # Dimensionality of the data

    # Initialize the parameters randomly
    w = np.zeros((n_samples, num_clusters))  # Responsibilities
    phi = np.random.dirichlet(np.ones(num_clusters), size=1)[0]  # Initial mixture weights (sum to 1)
    mean = np.random.uniform(low=np.min(X), high=np.max(X), size=(num_clusters, d))  # Initial means
    cov = [np.eye(d) for _ in range(num_clusters)]  # Initial covariance matrices (identity matrices)

    log_likelihoods = []

    for iteration in range(max_iters):
        w = e_step(num_clusters, X, phi, mean, cov)
        phi, mean, cov = m_step(num_clusters, X, w)

        # Calculate the log-likelihood
        log_likelihood = np.sum(np.log(np.sum([phi[j] * multivariate_gaussian_pdf(X, mean[j], cov[j])
                                               for j in range(num_clusters)], axis=0)))
        log_likelihoods.append(log_likelihood)

        # Check for convergence
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance:
            print(
                f"============================================ Converged in {iteration + 1} iterations, \nMean = {mean}, \nPhi = {phi}, \nCovariance = {cov}")
            break

        print(f"============================================ Iterations {iteration + 1}, \nMean = {mean}, \nPhi = {phi}, \nCovariance = {cov}")

    return phi, mean, cov, w


def main():
    num_cluster = 3

    # Mixing coefficients (probabilities for the latent variable z)
    phi = np.array([0.3, 0.4, 0.3])  # Probabilities should sum to 1

    # Means of the Gaussian components
    means = np.array([[0, 0], [5, 5], [10, 10]])  # k means for 2D data

    # Covariances of the Gaussian components
    covariances = [np.eye(2) * 0.5, np.eye(2) * 0.8, np.eye(2) * 0.3]  # 2D covariance matrices

    # Number of samples
    n_samples = 100

    # Generate the data
    X, z = generate_mixture_of_gaussians(n_samples, num_cluster, phi, means, covariances)

    em_algo(X, num_cluster, 1e-4)


if __name__ == "__main__":
    main()
