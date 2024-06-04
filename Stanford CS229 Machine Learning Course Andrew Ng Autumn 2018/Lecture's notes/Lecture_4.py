import numpy as np
import matplotlib.pyplot as plt

"""
n : Number of training example
k : The numbers of classes
logits : the output of the model
d : Number of parameters for each class prediction, equal to the number of inputs to the model
"""

n = 4
k = 3
d = 5
learning_rate = 0.01


# Create a target value matrix. For each example (row), one of the k classes is our target value
#y = np.zeros((n, k), dtype=int)
#for i in range(n):
#    y[i, np.random.randint(0, k)] = 1
y = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1],
              [0, 0, 1]])

print("Y : \n", y)

# The columns represent the number of parameters (equal to the number of input), the rows represent each class possible
#theta = np.random.uniform(0, 1, size=(k, d))
theta = np.ones((k, d))
# The columns represent the inputs, the rows represent each example
#x = np.random.uniform(0, 10, size=(n, d))
x = np.array([[1, 0.5, 0, 0, 0],
              [0, 0.5, 1, 0.5, 0],
              [0, 0, 0, 0.5, 1],
              [0, 0, 0, 0.5, 1]])
print("Theta : \n", theta)
print("X : \n", x)


def compute_logits(theta, x):
    logits = np.zeros((n, k), dtype=float)

    for example_index in range(n):  # The examples
        for class_index in range(k):  # The classes
            for input_index in range(d):  # The inputs / parameters
                logits[example_index][class_index] += x[example_index][input_index] * theta[class_index][input_index]

    return logits


# The logits are the outputs of the model before passing through the softmax function
# The columns are the prediction of the model for each class k, one row represent one example of the training set
logits = compute_logits(theta, x)
print("\nLogits : \n", logits)


def softmax(logits):  # One example of the training at a time
    assert(len(logits) == k)

    result = np.zeros(k)
    logits_sum = 0

    for i in range(k):
        logits_sum += np.exp(logits[i])

    for i in range(k):
        result[i] = np.exp(logits[i]) / logits_sum

    return result


""" Compute the softmax from a single output of the model """
def softmax_one_index(logits, model_index_output):  # One example of the training at a time
    assert(len(logits) == k)
    logits_sum = 0

    for i in range(k):
        logits_sum += np.exp(logits[i])

    return np.exp(logits[model_index_output]) / logits_sum


print("\nSoftmax result on the logits' first example (logits[0]) : ")
print(softmax(logits[0]))


""" 
Cross Entropy Loss 

The likelihood of one classe (from the k classes) is given by the k_th element of the softmax vector of the model output
So, the negative log-likelihood for each example of the training data in relation to the expected class's,
is the Loss Function.
Hence, for each example of the training data, we only compute the cross entropy loss on the k classe which is the 
correct classification class.

I think that, we do this because the output of the softmax function is a PDF (Probability Density Function), so the "error" of 
the model is : 1 - P(k = correct class), where P(k = correct class) is the output of the model after passing through 
the softmax function. P(k = correct class) should be as close as possible to 1, and we can measure the error with the 
deviation of the P(k = correct class) from 1.
"""
def cross_entropy_loss(logits, y):
    loss = 0

    for i in range(n):
        loss += - np.log(softmax_one_index(logits[i], np.argmax(y[i])))  # Only compute on : np.argmax(y[i]), which the right classification class

    return loss


print("\nCross Entropy Loss of the full dataset : \n", cross_entropy_loss(logits, y))


""" 
We obtain the gradient calculus by deriving the cross entropy loss with respect to the logits' of a class. 
"""
def compute_gradient(example_index):
    return softmax(logits[example_index]) - y[example_index]


print("\nGradients' of the first example :")
print(compute_gradient(0))


""" 
Stochastic Gradient Descent
We are using the same methode as above. We derive the Loss Function with respect to a certain theta. 
This give us the gradient calculus, which is the prediction of the model for a certain class in comparison with the 
expected value for this class (0 or 1)
"""
def stochastic_gradient_descent():
    global theta

    for example_index in range(n):
        gradient = compute_gradient(example_index)

        for class_index in range(k):
            theta[class_index] = theta[class_index] - learning_rate * gradient[class_index] * x[example_index]


print("\nTheta before : \n", theta)
for i in range(1000):
    stochastic_gradient_descent()
print("Theta after : \n", theta)

logits_after_trainig = compute_logits(theta, x)
print("\nLogits after training : \n", logits_after_trainig)

print("\nSoftmax result on the logits' after training : ")
for example_index in range(n):
    print(softmax(logits_after_trainig[example_index]))

print("\nCross Entropy Loss of the full dataset : \n", cross_entropy_loss(logits_after_trainig, y))

