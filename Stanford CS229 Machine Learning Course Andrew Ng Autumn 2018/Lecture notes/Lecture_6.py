import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load dataset for demonstration purposes
from sklearn.datasets import fetch_20newsgroups


"""
Naive Bayes or Multivariate Bernoulli event model
Bernoulli because each x is 0 or 1 (x_i represent the i_th word)

Naive Bayes is not very with other algorithm. But the advantages of naive bayes are that it is computationally very
efficient, and it is relatively quick to implement (doesn't require an iterative gradient descent).

To implement big and complex algorithm, the best way is to start with an easy to implement algorithm then look at the
example the algorithm is misclassifying. And only when you have the evidence that a particular upgrade could
significantly improve your algorithm performance, then you can spend a bunch of time solving this problem.
"""

"""
With Laplace smoothing
"""

number_of_classes = 2  # Here it is spam or non-spam


class NaiveBayesClassifier:
    def fit(self, X, y):
        # Number of documents
        n_docs = X.shape[0]
        # Number of words in the vocabulary
        n_words = X.shape[1]

        # Calculate phi_y
        self.phi_y = np.mean(y)

        # Calculate phi_j|y=1 and phi_j|y=0 with Laplace Smoothing
        self.phi_j_y1 = (X[y == 1].sum(axis=0) + 1) / (y.sum() + number_of_classes)
        self.phi_j_y0 = (X[y == 0].sum(axis=0) + 1) / ((n_docs - y.sum()) + number_of_classes)

    def predict(self, X):
        # Calculate log probabilities
        log_phi_j_y1 = np.log(self.phi_j_y1)
        log_phi_j_y0 = np.log(self.phi_j_y0)
        log_1_minus_phi_j_y1 = np.log(1 - self.phi_j_y1)
        log_1_minus_phi_j_y0 = np.log(1 - self.phi_j_y0)

        # Calculate log likelihoods
        log_likelihood_y1 = X @ log_phi_j_y1.T + (1 - X) @ log_1_minus_phi_j_y1.T
        log_likelihood_y0 = X @ log_phi_j_y0.T + (1 - X) @ log_1_minus_phi_j_y0.T

        # Calculate log posterior probabilities
        log_posterior_y1 = log_likelihood_y1 + np.log(self.phi_y)
        log_posterior_y0 = log_likelihood_y0 + np.log(1 - self.phi_y)

        # Predict y
        return (log_posterior_y1 > log_posterior_y0).astype(int)


categories = ['rec.autos', 'sci.space']  # Using two categories to simulate spam vs. non-spam
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
X_raw = newsgroups.data
y = (newsgroups.target == 1).astype(int)  # Arbitrarily consider 'sci.space' as spam (1)

# Convert text data to feature vectors
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(X_raw).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)

# Predict on the test set
y_pred = nb.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


""" 
Multinomial event model
"""


""" Support Vector Machine """
"""
Support Vector Machine or not as effective as Neural Network for many problems, but SVM or much more simple to use than 
NN. The isn't many parameters to customize.
"""