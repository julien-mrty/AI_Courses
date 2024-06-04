import numpy as np
import matplotlib.pyplot as plt

""" 
For the rest, I try to use :
i to refer to the training example number
j to refer to the input / parameter number 
"""

"""
CHAPTER 1
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


m = 2
n = 5

# Create a random matrix with shape (n, m + 1), the rows are the examples of the training set
# and the columns are the inputs of each example
x = np.random.uniform(0, 10, size=(n, m + 1))

# Set the first row to all ones because of the intercept term
x[:, 0] = 1
y = np.random.uniform(0, 10, size=(n, 1))
theta = np.ones(m + 1)


print("======================================================== Initial values : ")
print("X = ", x)
print("y = ", y)
print("theta = ", theta)

learning_rate = 0.01


""" h(x) : The hypothesis of the model """
""" Here x is one example from the training set """
def h(x):
    global theta, m

    assert(len(x.shape) == 1)  # We make sure that x is unidimensional (one example of the training set)

    h = 0
    for j in range(m + 1):
        h += theta[j] * x[j]

    return h


""" Here j is the number of the input from the example of the training set """
def h(x, j):
    global theta

    assert(len(x.shape) == 1)  # We make sure that x is unidimensional (one example of the training set)

    return theta[j] * x[j]


""" J() : The cost function """
def J():
    global x, y, m

    J = 0
    for i in range(n):
        J += 1/2 * (h(x[i]) - y[i]) ^ 2

    return J


""" Gradient descent """
""" 
We perform the parameters' update through Stochastic Gradient Descent algorithm (also Incremental Gradient Descent)
For each exemple of the training set as input, we update the parameters
Other method : Batch Gradient Descent, run through multiple or all training examples before updating the parameters

We want to choose theta so as to minimize J() 
theta-j depends on, the derivative of the cost function J() with respect of theta-j, multiplied by the learning rate
When the calculations are performed, you end up with the following calculus
"""
def stochastic_gradient_descent():
    global learning_rate, theta, x, y

    for i in range(n):  # We browse the list of examples
        for j in range(m + 1):  # We update each parameter
            theta[j] = theta[j] + learning_rate * (y[i] - h(x[i])) * x[i][j]


""" 
Locally weighted linear regression 
We still want to fit theta to minimize the following formula (our new loss function)

To simplify let's say that, x have only input dans no intercept term. 
We want to make a prediction around one value of x_
"""
def define_new_values_for_experiment(number_of_points):
    # Define the range of x values
    x_ = np.linspace(0, 2 * np.pi, number_of_points)  # 100 points from 0 to 2*pi

    # Define the y values as the sine of x
    y_ = np.sin(x_)

    theta_ = 1  # Only one input

    # Create the plot
    plt.plot(x_, y_)

    # Add title and labels
    plt.title('Sine Wave')
    plt.xlabel('x values')
    plt.ylabel('sin(x)')

    # Show the plot
    plt.show()

    return x_, y_, theta_


def locally_weighted_linear_regression_cost(x_prediction, tau):
    cost = 0

    number_of_examples = 100

    x, y, theta = define_new_values_for_experiment(number_of_examples)

    for i in range(number_of_examples):
        cost += w(i, x_prediction, x, tau) * (y[i] - (theta * x[i])) ** 2

    return cost


def w(i, x_prediction, x, tau):
    return np.exp(- (x[i] - x_prediction) ** 2 / (2 * tau ** 2))


print()
#print("LWR cost function : ", locally_weighted_linear_regression_cost(np.pi, 1))
"""
As I understand it, this cost function will be used to update the weights to fit a prediction around a certain point.
Tau allow to control how quickly the weight of a training example falls off with distance
"""



"""
CHAPTER 2 : 
Classification and logistic regression 
"""

# Redefine y because our new target is [0, 1]
# y = np.random.rand(n, 1)
y = [0, 0.2, 0.4, 0.6, 0.8]
x = np.array([[1, 0, 0],
              [1, 0.15, 0.25],
              [1, 0.35, 0.45],
              [1, 0.55, 0.65],
              [1, 0.75, 0.85]])

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
