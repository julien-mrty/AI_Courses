"""
III) Generalization and regularization
Chapter 9
Regularization
"""

"""
Definition of bias in ML : A learning algorithm as very strong preconceptions that the data could be fit
by a certain family (ex : linear function or linear model family's).

Definition of variance in ML : An algorithm which have a high variance, will fit a totally different function if the
data used to train the algorithm is slightly different. There is a lot of variability in the prediction of the algorithm
"""

"""
Regularization

Regularization involves adding a term to the cost/loss function. In most cases the regularizer is non negative function
and is purely a function of the parameters theta. It is also typically chosen to be some measure of the complexity of 
the model. Thus we aim to find a model that fit the data and have a small model complexity. The balance between the two 
objectives is controlled by the regularization parameter λ. When λ = 0, the regularized loss is equivalent to the 
original loss. When λ is a sufficiently small positive number, minimizing the regularized loss is effectively minimizing 
the original loss with the regularizer as the tie-breaker. When the regularizer is extremely large, then the original 
loss is not effective (and likely the model will have a large bias.)

In addition to simplifying the models, regularization can also impose biases or structure on the model parameters. 
Imposing additional structure of the parameters narrows our search space and makes the complexity of the model family 
smaller. Thus model's tend to better generalization. On the other hand, imposing additional structure may risk
increasing the bias.
"""


"""
Model selection via cross validation

See the part "Size of the different sets" bellow.
You must test your model on a specific test set to be sure that your model is able to generalize from its training 
examples. Thus you chose the model that generalize best and have the lowest test error.

Here’s an algorithm that works better. In hold-out cross validation (also called simple cross validation), we do the 
following:
    
    1. Randomly split S into Strain (say, 70% of the data) and Scv (the remaining 30%). Here, Scv is called the hold-out 
    cross validation set.
    2. Train each model Mi on Strain only, to get some hypothesis hi.
    3. Select and output the hypothesis hi that had the smallest error ˆεScv (hi) on the hold out cross validation set. 
    (Here εScv (h) denotes the average error of h on the set of examples in Scv.) The error on the hold out validation 
    set is also referred to as the validation error.

Optionally, step 3 in the algorithm may also be replaced with selecting the model Mi according to arg mini εˆScv (hi), 
and then retraining Mi on the entire training set S. (This is often a good idea, with one exception being learning 
algorithms that are be very sensitive to perturbations of the initial conditions and/or data. For these methods, Mi 
doing well on Strain does not necessarily mean it will also do well on Scv, and it might be better to forgo this 
retraining step.)


The issue of that method is that it "waste" 30% of the data. Even if we were to take the optional step of retraining
the model on the entire training set, it’s still as if we’re trying to find a good model for a learning problem in which 
we had 0.7n training examples, rather than n training examples.
While this is fine if data is abundant and/or cheap, in learning problems in which data is scarce (consider a problem 
with n = 20, say), we’d like to do something better. Here is a method, called k-fold cross validation, that holds out 
less data each time:

    1. Randomly split S into k disjoint subsets of m/k training examples each. Lets call these subsets S1, . . . , Sk.
    2. For each model Mi, we evaluate it as follows:
        For j = 1, . . . , k
            Train the model Mi on S1 ∪ · · · ∪ Sj−1 ∪ Sj+1 ∪ · · · Sk (i.e., train on all the data except Sj ) to get some 
            hypothesis hij.
                Test the hypothesis hij on Sj, to get ˆεSj (hij ).
        The estimated generalization error of model Mi is then calculated as the average of the ˆεSj (hij )’s (averaged 
        over j).
    3. Pick the model Mi with the lowest estimated generalization error, and retrain that model on the entire training 
    set S. The resulting hypothesis is then output as our final answer.
    
A typical choice for the number of folds to use here would be k = 10. While the fraction of data held out each time is 
now 1/k—much smaller than before—this procedure may also be more computationally expensive than hold-out cross 
validation, since we now need train to each model k times.
"""


"""
Bayesian Statistics and Regularization

Overfitting can be combated using Bayesian statistics and regularization. Initially, we discussed parameter fitting 
using Maximum Likelihood Estimation (MLE), which views parameters (𝜃) as unknown but fixed, consistent with the 
frequentist perspective.
Bayesian Approach : Treats parameters (𝜃) as random variables with a prior distribution 𝑝(𝜃), reflecting our prior 
beliefs about these parameters.

Check lecture notes for details and calculus. Not simply summarizable.
"""


"""
Size of the different sets.

To choose the size of the training, test and development set. It is not always relevant to divide the dataset in :
80%, 10%, 10%. It maybe is if you have a reasonable size dataset (let's say from 500 to 10.000 data). But when your 
dataset is composed of 10.000.000 data, there is not useful to to use 2.000.000 data to test the model. When choosing 
the sizes of your test and dev sets, you have to use the minimum size to give you relevant difference performance of 
your model. For example, if you want to improve a model prediction by 10%, a model improvement of 0.5% is negligible.
But I you are trying to improve your model by 0.1%, you need way more test data to be sure that your model really 
improved by 0.1%. For example, improve click rate by 0.1% allows you to significantly increase your income. To do so, 
you may improve your model on 10 different lever by 0.01% to achieve your 0.1% improvement on the overall model. To be 
sure of your 0.01% improvement you may need millions of data. But in most cases it is irrelevant.
"""