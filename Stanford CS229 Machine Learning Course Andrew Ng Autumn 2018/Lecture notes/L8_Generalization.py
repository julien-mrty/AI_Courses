import numpy as np


"""
III) Generalization and regularization
Chapter 8
Generalization
"""


"""
Definition of bias in ML : A learning algorithm as very strong preconceptions that the data could be fit
by a certain family (ex : linear function or linear model family's).

Definition of variance in ML : An algorithm which have a high variance, will fit a totally different function if the
data used to train the algorithm is slightly different. There is a lot of variability in the prediction of the algorithm
"""


"""
- Underfitting and bias : 
A model that is not able represent the data, even if the amount of data would be infinite, is underfitting. We define 
this as a bias. This model has a large training and testing error.
For example a linear model that try to fit a quadratic function.


- Overfitting and variance :
A model that have a very small margin of error on training examples but, has a very large one on the tests examples, is 
unable to generalize. The model is overfitting.
For example a 5th degree polynomial model that try to fit a quadratic function.
The 5th degree polynomial is in practice able to fit a quadratic function with a large amount of data. If the 3rd, 4th
and 5th degree = 0, it is still a 5th degree polynomial that is capable of capturing the structure of the data.
The failure of fitting, by a "too complex model" (overfitting), can be captured by another component of the test error 
called variance of a fitting model procedure. With "too complex model", there is a large risk that the model is fitting
patterns in the data that happened to be present in our small, finite training set, but that do not reflect the wider
pattern of the relationship between x and y. These spurious patterns are mostly due to observation noise. The model
that fit this spurious patterns ended with a larger test error. In this case, the model has a large variance. The 
spurious patterns are specific to the randomness of the noise and input in a dataset. Thus, they are specific to each
dataset.
Another to clue as to whether the model is overfitting is when you train your model on different training set that have the
same distribution, and the model has very different predictions on the test set.

Often, there is a tradeoff between bias and variance. If our model is too “simple” and has very few parameters, then it 
may have large bias (but small variance), and it typically may suffer from underfitting. If it is too “complex” and has 
very many parameters, then it may suffer from large variance (but have smaller bias), and thus overfitting. 
"""


"""
Mathematical decomposition of the bias variance tradeoff.

Check the lecture notes for more details and mathematical formula

The variance term captures how the random nature of the finite dataset introduces errors in the learned model. It 
measures the sensitivity of the learned model to the randomness in the dataset. It often decreases as the size of the 
dataset increases. There is nothing we can do about the first term σ2 as we can not predict the noise ξ by definition.
Finally, we note that the bias-variance decomposition for classification is much less clear than for regression 
problems. There have been several proposals, but there is as yet no agreement on what is the “right” and/or the most 
useful formalism
"""
