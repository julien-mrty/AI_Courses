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
def stochastic_gradient_descent(theta, X):
    for index_example in range(n):
        phi_x = Phi_X[index_example]
        theta += learning_rate * (y[index_example] - (theta @ phi_x)) * phi_x

    return theta


print("\nTheta : ", theta)
theta = stochastic_gradient_descent(theta, X)
print("Theta after SGD : ", theta)


# Beta, set of coefficient for kernel trick
beta = np.zeros(n)


""" LMS with the kernel trick """
""" In the following, phi contains all monomials of x with degree <= 3 """
def beta_update_with_kernel_trick(beta, X):
    K = compute_kernel_matrix(X)

    for index_example in range(n):
        beta[index_example] += + learning_rate * (y[index_example] - (K[index_example] @ beta))

    return beta


""" 
Compute the kernel matrix 
The kernel matrix correspond to the feature map phi of two different variables
Example : K = < phi(x), phi(z) > 

The polynomial kernel used here only works for the phi function currently used (all monomials of x with degree <= 3)
"""
def polynomial_kernel(x, z):
    # Compute the polynomial kernel value between vectors x and z.

    dot_product = np.dot(x, z)
    return 1 + dot_product + dot_product**2 + dot_product**3


def compute_kernel_matrix(X):
    # Compute the kernel matrix K for a given set of input vectors X.

    K = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            K[i, j] = polynomial_kernel(X[i], X[j])

    return K


print("\nBeta : ", beta)
beta = beta_update_with_kernel_trick(beta, X)
print("Beta after kernel trick : ", beta)


def predict(x):
    predictions = np.zeros(n)

    for i in range(n):
        k = np.array([polynomial_kernel(x[i], x_j) for x_j in x])
        predictions[i] = np.dot(beta, k)

    return predictions


print("Prediction : ", predict(X))


""" Support Vector Machine """
"""
Support Vector Machine or not as effective as Neural Network for many problems, but SVM is much more simple to use than 
NN. There isn't many parameters to customize like in NN, it is more straightforward.
"""
