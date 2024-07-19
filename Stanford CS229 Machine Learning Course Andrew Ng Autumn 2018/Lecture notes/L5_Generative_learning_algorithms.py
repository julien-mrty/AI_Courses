import numpy as np


"""
I) Supervised learning
Chapter 4
Generative Learning Algorithm
"""


"""
To maximize the log-likelihood for phi :
Take the log of the likelihood
Take the derivative 
Set the derivative = 0
Solve for the values of the parameters that maximize the function
"""

"""
If you don't know the nature the distribution of your data, Logistic Regression is the best answer.
If you data follow a Poisson distribution but you assume that it was Gaussian and use a Gaussian Discriminant analysis,
the model will perform poorly.

Key high level principles :
- If you make weaker assumptions as a logistic regression, your algorithm will be more robust to modeling assumptions.
- If you have a very small dataset, use a model that makes more assumptions will allow the model to do better
- If you have many data, the safe choice is to use weaker assumptions algorithm (such as a logistic regression), because 
with more data you can overcome weaker assumptions (being less precise about the choice of the algorithm)

Gaussian Discriminant Algorithm is quite computational efficient.
"""


"""
Gaussian Discriminant Analysis model
"""

n = 6  # Number of examples

# Each line is one example, # of inputs = d
x = np.array([[1.2, 0.7],
              [1.3, 0.9],
              [1.1, 0.65],
              [5.1, 5.1],
              [4.9, 5.2],
              [5.2, 4.8],])

y = np.array([0, 0, 0, 1, 1, 1])


phi = 0.8
d = 2
mu0 = np.array([1, 1])

mu1 = np.array([5, 5])

# Arbitrarily chosen values
covariance = np.array([[1, 0.8],
                       [0.8, 1]])


def P_y(y):
    return (phi ** y) * (1 - phi) ** (1 - y)


# np.multiply : multiply matrix by scalar
# np.matmul : multiply matrices


def P_xy(x, y):
    if y == 0:
        return ((1 / (np.multiply( (2 * np.pi) ** (d/2), np.linalg.det(covariance) ** (1/2) ))) *
            np.exp(np.matmul(
                   np.matmul(
                   np.multiply((-1 / 2), np.transpose(np.subtract(x, mu0))),
                        np.linalg.inv(covariance)),
                            np.subtract(x, mu0))))
    elif y == 1:
        return (1 / (np.multiply( (2 * np.pi) ** (d/2), np.linalg.det(covariance) ** (1/2) ))) * np.exp(
            np.matmul(np.matmul(np.multiply((-1 / 2), np.transpose(np.subtract(x, mu1))), np.linalg.inv(covariance)),
                      np.subtract(x, mu1)))
    else:
        print("P(x, y), y wrong value : ", y)


""" The loglikelihood of the data with respect to the parameters """
def loglikelihood(x, y):
    loglikelihood = 1

    for example_index in range(n):
        loglikelihood *= P_xy(x[example_index], y[example_index]) * P_y(y[example_index])

    return np.log(loglikelihood)


print("Loglikelihood before parameters' adjustments : ", loglikelihood(x, y))


def maximize_loglikelihood_phi():
    global phi
    phi = np.mean(y == 1)


def maximize_loglikelihood_mu():
    global mu0, mu1

    mu0 = np.zeros(d)
    cpt_y0 = 0

    mu1 = np.zeros(d)
    cpt_y1 = 0

    for example_index in range(n):
        for input_index in range(d):
            mu = x[example_index]

        if y[example_index] == 0:
            cpt_y0 += 1
            mu0 += mu
        else:
            cpt_y1 += 1
            mu1 += mu

    mu0 /= cpt_y0
    mu1 /= cpt_y1


def maximize_loglikelihood_covariance():
    global covariance

    covariance = np.zeros((d, d))
    for example_index in range(n):
        if y[example_index] == 0:
            covariance += np.outer(x[example_index] - mu0, x[example_index] - mu0)
        elif y[example_index] == 1:
            covariance += np.outer(x[example_index] - mu1, x[example_index] - mu1)


def maximize_loglikelihood():
    maximize_loglikelihood_phi()
    print("phi : ", phi)

    maximize_loglikelihood_mu()
    print("mu0 : ", mu0)
    print("mu1 : ", mu1)

    maximize_loglikelihood_covariance()
    print("covariance : ", covariance)


maximize_loglikelihood()

"""
The maximisation of the loglikelihood of the parameters is still high. The objective here, is to have a probability as 
close to 1 as possible at the output of our model (perfect prediction). So le log of our prediction should 
be log(1) = 0. Here : -1,35
"""
print("\nloglikelihood after maximization : ", loglikelihood(x, y))
