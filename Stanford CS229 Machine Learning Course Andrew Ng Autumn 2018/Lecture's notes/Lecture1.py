import numpy as np

"""
Chapter 1
Linear regression
"""

"""
Hypothesis made the model : h
Parameters (weights) : theta
Inputs / features : x
Output / target variable : y
Number of inputs : m
Intercept term : x0 = 1
Number of training example : n
"""

m = 3
n = 10
x = np.random.rand(m, n)
y = [1, 10.5, 11.7, 16.3]
theta = [1, 1, 1, 1]


learning_rate = 0.01


""" h(x) : The hypothesis of the model """
def h(x):
    global theta, m

    h = 0
    for i in range(m + 1):
        h += theta[i] * x[i]

    return h


def h(x, i):
    global theta

    return theta[i] * x[i]


""" J(theta) : The cost function """
def J(theta):
    global x, y, m

    J = 0
    for i in range(m + 1):
        J += 1/2 * (h(x, i) - y[i]) ^ 2

    return J


""" Gradient dscent """
def gradient_descent(theta, i):
    global learning_rate, x, y

    theta[i] = theta[i] + learning_rate * (y[i] - h(x, i)) * x[i]


def gradient_descent(theta):
    global learning_rate, x, y

    for i in range(m + 1):
        gradient_descent(theta, i)