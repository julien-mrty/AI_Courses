import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

"""
Multinomial event model
"""

# Define number of classes
number_of_classes = 2  # Here it is spam or non-spam

# Generate data
np.random.seed(42)  # For reproducibility
n_samples = 1000  # Number of samples
number_of_vocab_word = 200  # Maximum number of features (words in vocabulary)
number_feature_per_sample = np.random.randint(3, 10, n_samples)

# Half samples with output 1 and other half with output 0
n_samples_class_1 = n_samples // 4
n_samples_class_0 = (3 * n_samples) // 4
print(n_samples_class_1)
print(n_samples_class_0)

# Generate features for class 1 and class 0
X_class_1 = []
X_class_0 = []
cross = 0.1  # Cross between examples

print((number_of_vocab_word - number_of_vocab_word * cross) / 2)
print((number_of_vocab_word + number_of_vocab_word * cross) / 2)

for i in range(n_samples_class_1):
    num_features_1 = number_feature_per_sample[i]
    features_1 = np.random.randint(int((number_of_vocab_word - number_of_vocab_word * cross) / 2), number_of_vocab_word,
                                   size=num_features_1)
    X_class_1.append(features_1)

for i in range(n_samples_class_0):
    num_features_0 = number_feature_per_sample[n_samples_class_1 + i]
    features_0 = np.random.randint(0, int((number_of_vocab_word + number_of_vocab_word * cross) / 2),
                                   size=num_features_0)
    X_class_0.append(features_0)

# Combine the features into a single list
X = X_class_1 + X_class_0

# Convert the list to an array with objects (each row can have a different length)
X = np.array(X, dtype=object)

# Create labels
y = np.array([1] * n_samples_class_1 + [0] * n_samples_class_0)


class EventModelClassifier:
    def __init__(self):
        self.phi_y = 0
        self.phi_j_y0 = 0
        self.phi_j_y1 = 0

    def fit(self, X, y):
        # Number of emails
        n_emails = len(X)
        # Size of vocabulary
        vocabulary_size = np.max([np.max(doc) for doc in X]) + 1  # size of vocabulary

        words_appearance_y0 = np.zeros(vocabulary_size + 1)
        words_appearance_y1 = np.zeros(vocabulary_size + 1)

        total_word_y0 = 0
        total_word_y1 = 0

        # Calculate phi_y
        self.phi_y = np.mean(y)

        for index_email in range(n_emails):
            if y[index_email] == 0:
                total_word_y0 += len(X[index_email])

                for index_word in range(len(X[index_email])):
                    words_appearance_y0[X[index_email][index_word]] += 1
            else:
                total_word_y1 += len(X[index_email])

                for index_word in range(len(X[index_email])):
                    words_appearance_y1[X[index_email][index_word]] += 1

        # Calculus with Laplace Smoothing
        self.phi_j_y0 = (words_appearance_y0 + 1) / (total_word_y0 + vocabulary_size)
        self.phi_j_y1 = (words_appearance_y1 + 1) / (total_word_y1 + vocabulary_size)

    def predict(self, X_test):
        n_test = len(X_test)
        predictions = np.zeros(n_test)

        for i in range(n_test):
            log_prob_1 = np.log(self.phi_y)
            log_prob_0 = np.log(1 - self.phi_y)

            for word_index in range(len(X_test[i])):
                log_prob_1 += np.log(self.phi_j_y1[X_test[i][word_index]])
                log_prob_0 += np.log(self.phi_j_y0[X_test[i][word_index]])

            if log_prob_1 > log_prob_0:
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions

    def predict_mul(self, X_test):
        n_test = len(X_test)
        predictions = np.zeros(n_test)

        for i in range(n_test):
            prob_1 = self.phi_y
            prob_0 = 1 - self.phi_y

            for word_index in range(len(X_test[i])):
                prob_1 *= np.log(self.phi_j_y1[X_test[i][word_index]])
                prob_0 *= np.log(self.phi_j_y0[X_test[i][word_index]])

            if prob_1 > prob_0:
                predictions[i] = 1
            else:
                predictions[i] = 0

        return predictions



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize and train the model
model = EventModelClassifier()
model.fit(X_train, y_train)

print("model.phi_y : ", model.phi_y)
print("Sum y0 : ", sum(model.phi_j_y0))
print("Sum y1 : ", sum(model.phi_j_y1))

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy (using addition to predict) : {accuracy * 100:.2f}%')


# Predict on the test set
y_pred_mul = model.predict_mul(X_test)

# Calculate accuracy
accuracy2 = accuracy_score(y_test, y_pred_mul)
print(f'Accuracy (using multiplication to predict) : {accuracy2 * 100:.2f}%')

"""
When I compute the probabilities directly (predict_mul), the model tends to encounter numerical underflow issues, 
causing very small numbers to round to zero. This is why the accuracy of predict_v2 method is significantly lower.

Conversely, when I use log probabilities (predict), I convert the product of probabilities into a sum of log 
probabilities. This approach is numerically stable and avoids underflow problems.
"""