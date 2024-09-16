import numpy as np
from itertools import product


"""
I) Supervised learning
Chapter 5
Kernel methods
"""


np.random.seed(42)  # For reproducibility

max_degree = 3  # The maximum monomial degree of x


def generate_random_x(d, n):
    return np.random.uniform(0, 10, (n, d))


def compute_monomials(x):
    """
    Generate all monomials up to the third degree for a 3D vector x.

    Args:
    x (list or np.ndarray): A 3D vector [x1, x2, x3]

    Returns:
    np.ndarray: Array containing all the monomials including repetitions.
    """
    x = np.asarray(x)
    d = len(x)

    # Generate monomials for each degree
    monomials = []
    for degree in range(max_degree + 1):
        for exponents in product(range(d), repeat=degree):
            if not exponents:
                monomial = 1
            else:
                monomial = np.prod([x[i] for i in exponents])
            monomials.append(monomial)

    return np.array(monomials)


def compute_phi_for_examples(X):
    # Compute the dimension of the feature vector for one example
    d = X.shape[1]

    # Number of features
    num_features = 1 + d + d ** 2 + d ** 3
    print("Number of output feature for phi : ", num_features)

    # Initialize a matrix to store the feature vectors for all examples
    Phi = np.zeros((X.shape[0], num_features))

    # Compute phi(x) for each example x in X
    for i, x in enumerate(X):
        Phi[i, :] = compute_monomials(x)

    return Phi


# Example usage
d = 3  # Dimension of each x
n = 5  # Number of examples

# Generate n random vectors of dimension d
X = generate_random_x(d, n)

# Compute the feature vectors for all examples
Phi_X = compute_phi_for_examples(X)

print("\nX (input examples) :")
print(X)
print("\nPhi(X)[0] (feature vectors, only first raw) :")
print(Phi_X[0])


# The class corresponding to each example
y = np.random.uniform(0, 10, size=n)

# The parameters theta
theta = np.zeros(len(Phi_X[0]))

# The learning rate
learning_rate = 0.01


""" 
Least Mean (LMS) Squares with features using a derivative of Stochastic Gradient Descent
"""
class KernelLMS:
    def __init__(self, alpha, degree=3):
        self.alpha = alpha
        self.degree = degree
        self.beta = None
        self.K = None

    def polynomial_kernel_old(self, x, z):
        # Compute the polynomial kernel value
        return (1 + np.dot(x, z)) ** self.degree

    def polynomial_kernel(self, x, z):
        # Compute the polynomial kernel value between vectors x and z.

        dot_product = np.dot(x, z)
        return 1 + dot_product + dot_product ** 2 + dot_product ** 3

    def compute_kernel_matrix(self, X):
        # Compute the kernel matrix K for a given set of input vectors X.
        self.K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                self.K[i, j] = self.polynomial_kernel(X[i], X[j])

    def fit(self, X, y):
        n_samples = X.shape[0]

        # Compute the kernel matrix
        self.compute_kernel_matrix(X)

        # Initialize beta coefficients
        self.beta = np.zeros(n_samples)

        # Iterative update
        for i in range(n_samples):
            prediction_error = y[i] - np.dot(self.K[i], self.beta)
            self.beta[i] += self.alpha * prediction_error

    def predict(self, X):
        if self.beta is None or self.K is None:
            raise ValueError("The model has not been trained yet.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            k = np.array([self.polynomial_kernel(X[i], x_j) for x_j in X])
            predictions[i] = np.dot(self.beta, k)

        return predictions


# Create the Kernel LMS model
model = KernelLMS(alpha=0.01)

# Fit the model
model.fit(X, y)

# Predict
predictions = model.predict(X)
print("\nPredictions : ", predictions)
