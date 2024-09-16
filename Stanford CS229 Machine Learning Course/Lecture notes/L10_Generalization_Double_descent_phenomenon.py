"""
III) Generalization and regularization
Chapter 8
Generalization
"""


"""
Double descent phenomenon
"""


"""
Model-wise double descent :

General case : as we increase the model complexity, the test error first decreases then increases.
The double descent phenomenon is a second decrease after the first increase of the test error. We generally observe
a first increase when the model size is large enough to fit all the training data very well, and then decrease again
in the so-called over-parameterized regime, where the number of parameters is larger than the number of data points.

We observe the double descent phenomenon in many cases.
The conclusion to be drawn is that we should not hesitate to scale our model and experimenting with the 
over-parametrized models because the test error may decrease again to a level even smaller than the previous lowest 
point. In many cases, larger over-parameterized models always lead to a better test performance, meaning there won't 
be a second ascent after the second descent.
"""

"""
Sample-wise double descent :

At first glance, we would expect that more training example always lead to smaller test errors. However, recent work 
observes that the test error is not monotonically decreasing as we increase the sample size. Instead, the test error 
decreases, and then increases and peaks around when the number of examples (denoted by n) is similar to the number of 
parameters (denoted by d), and then decreases again. We refer to this as the sample-wise double descent phenomenon. To
some extent, sample-wise double descent and model-wise double descent are essentially describing similar phenomena|the
test error is peaked when n ~ d.
"""

"""
Explanation and mitigation strategy :

The "double descent" phenomenon in machine learning, where test error peaks when the number of samples n is close to the 
number of model parameters d. The observed peak in test error when n is close to d suggests that current training 
algorithms are not optimal in this regime, and a better strategy might be to use fewer samples to avoid the error peak. 
The phenomenon can be mitigated by optimally tuning regularization, which improves test error when n > d and reduces 
both sample-wise and model-wise double descent.

Over-parameterized models (where d > n) generalize well despite having more parameters than samples. It suggests that 
commonly-used optimizers like gradient descent provide implicit regularization, leading to better test performance even 
without explicit regularization. For linear models, this implicit regularization often results in selecting a minimum 
norm solution, which works well when d > n but not when n > d.

The document notes that double descent has been primarily observed when model complexity is measured by the number of 
parameters, but this might not always be the best measure. For example, when complexity is measured by the norm of the 
learned model, the double descent phenomenon does not occur. This suggests that understanding the correct complexity 
measure, especially for deep neural networks, remains an active area of research.
"""
