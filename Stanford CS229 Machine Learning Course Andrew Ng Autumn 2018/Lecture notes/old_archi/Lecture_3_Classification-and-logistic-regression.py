import numpy as np
import matplotlib.pyplot as plt

"""
I) Supervised learning
CHAPTER 2 :
Classification and logistic regression
"""

m = 2
n = 5

# Create a random matrix with shape (n, m + 1), the rows are the examples of the training set
# and the columns are the inputs of each example
x = np.random.uniform(0, 10, size=(n, m + 1))

# Set the first row to all ones because of the intercept term
x[:, 0] = 1
y = np.random.uniform(0, 10, size=(n, 1))
theta = np.ones(m + 1)

# Redefine y because our new target is [0, 1]
# y = np.random.rand(n, 1)
y = [0, 0.2, 0.4, 0.6, 0.8]
x = np.array([[1, 0, 0],
              [1, 0.15, 0.25],
              [1, 0.35, 0.45],
              [1, 0.55, 0.65],
              [1, 0.75, 0.85]])


print("======================================================== Initial values : ")
print("X = ", x)
print("y = ", y)
print("theta = ", theta)

learning_rate = 0.01

print("\n New Y = ", y)


""" 
New form of the model hypothesis 
h(x) = g(theta-T * x)
The output of our new hypotheses is between 0 and 1
The goal here is to have a classification approach
"""
def g(theta, x):
    assert(len(x.shape) == 1)  # We make sure that x is unidimensional (one example of the training set)
    assert(len(x) == len(theta))

    return 1 / (1 + np.exp(- z(theta, x)))


# z() is the same calculus as h() but z() isn't use as the hypothesis
def z(theta, x):
    z = 0

    for j in range(m + 1):
        z += theta[j] * x[j]

    return z

print("\nX[0] : ", x[0])
print("Theta : ", theta)
print("Model hypothesis for classification between 2 classes : ", g(theta, x[0]))


"""
We want to maximize the likelihood of theta 
The new value of theta depends on the learning rate and the derivative of the log likelihood of theta. We want to 
maximize the log likelihood so we use + here.

Finally, when performing the calculus we obtain the Stochastic Gradient Ascent rule, wich we use this way :
"""
def stochastic_gradient_ascent():
    for i in range(n):
        for j in range(m + 1):
            theta[j] = theta[j] + learning_rate * (y[i] - g(theta, x[i])) * x[i][j]


print("\nUpdating weights....")
for i in range(10000):
    stochastic_gradient_ascent()

print("NEW Model hypothesis : ", g(theta, np.array([1, 0.75, 0.85])))
print("NEW theta : ", theta)
