"""
I) Supervised learning
Chapter 6
Support Vector Machine
"""


"""
Support Vector Machine is not as effective as Neural Network for many problems, but SVM is much more simple to use than 
NN. There isn't many parameters to customize like in NN, it is more straightforward.
"""

"""
Optional reading, I followed the lecture but just skimmed the notes.
"""

"""
The decision boundary is set by our model.
I say points because in the example it is a 2D plan, thus the boundary is a line.
The closer the point is to the decision boundary, lower is our confidence in the prediction. Conversely, if the point is
far from the decision boundary, our confidence in the prediction of our model is strong.

The goal here, is to set our decision boundary in a way to have our two different classes' points as far as the decision
boundary as possible.

It is possible to map data into higher-dimensional spaces to find better decision boundaries. 

- Linear Decision Boundaries :
In the original feature space, SVM tries to find a linear decision boundary (hyperplane) that separates different 
classes with the maximum margin.

- Non-linear Decision Boundaries :
Many problems are not linearly separable in their original feature space. SVM addresses this by mapping the input 
features into a higher-dimensional space where a linear separation is possible.

To achieve this higher-dimensional mapping efficiently, SVMs use the kernel trick. This allows SVMs to compute the 
decision boundary in the higher-dimensional space without explicitly performing the transformation. This allows SVMs to 
handle complex, non-linearly separable data efficiently by transforming it into a space where a linear separator can be 
found, and then translating this separator back into a non-linear boundary in the original space.
"""