"""
III) Generalization and regularization
Chapter 8
Generalization
"""


"""
Bias-Variance tradeoff 


How to fight variance ?
1) Increase the number of data
2) Add regularization


Example : 
If you have a low bias high variance algorithm, you can add regularization. By adding regularization, you may end up
with a small (a bit more than low) bias, low variance algorithm.
"""

"""
Model prediction error decomposition

Check lecture's 10 screenshot.

You can decompose the model prediction error in 3 parts :
1 - The Bayes error / Irreducible error. The difference between the perfect hypothesis and the best possible hypothesis.
2 - Approximation error (due to the class of models' chosen). It is the difference the best hypothesis of our model and 
the best hypothesis possible.
3 - Estimation error (due to limited data). Given the data that we have, it is the difference between the best 
hypothesis of our model and the average hypothesis of our model.

=== The average error of our model = Estimation error + Approximation error + Irreducible error ===
The average error of our model is also called the generalization error.

The irreducible error can't be decreased. Then you make some decisions about what class/type of models you want to use
(neural network, logistic regression, etc...), this is the approximation error. Finally you work with limited data and
possible nuances of your algorithm that causes the estimation error.


- Estimation error = Estimation variance + Estimation bias 
- Variance = Estimation variance
- Bias = Estimation bias + Approximation error

Then we get :
=== The average error of our model = Variance + Bias + Irreducible error ===

The bias globally try to capture why, the average hypothesis of our model is far from the best possible hypothesis.
The variance almost allways due to having small dataset.


How to fight high bias ?

Make H bigger (check lecture's screenshot) -> reduce your bias but increase your variance.
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
