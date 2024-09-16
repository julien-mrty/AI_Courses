"""
III) Deep learning
Chapter 7
Deep learning
"""


"""
Neural networks
"""


"""
Reminder :
Difference between Gradient Descent dans Stochastic Gradient Descent. With Gradient Descent, the parameters' update is 
performed after reviewing the whole dataset, it is also called Batch Gradient Descent. With Stochastic Gradient Descent,
the parameters' update is performed after each cost/loss function calculus on a single example of the dataset. There's also 
the Mini-Batch Gradient Descent which is performed on a small subset of the dataset.
"""


"""
With Mini-Batch Gradient Descent, bigger is the batch, faster the training is. Lower is the batch, better are the
performances of the algorithm. Generally we choose the batch size, as the maximum batch size that fit the GPU memory 
(it stays relatively low). 
"""