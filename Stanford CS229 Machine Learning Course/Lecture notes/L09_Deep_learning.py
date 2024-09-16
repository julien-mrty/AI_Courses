import numpy as np


"""
III) Deep learning
Chapter 7
Deep learning
"""


"""
Backpropagation
"""

"""
Here I use what I did previously and add the backpropagation
"""


np.random.seed(42)  # For reproducibility


def ReLU(z):
    return np.maximum(z, 0)


def ReLU_prime(z):
    return np.where(z < 0, 0, 1)  # Return 1 if z > 0 or 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


class Layer:
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, last_layer):
        self.weights = np.random.rand(num_neurons_current_layer, num_neurons_previous_layer)
        # self.weights = np.random.uniform(-1, 1, (num_neurons, num_neurons_previous_layer))
        self.bias = np.random.uniform(-1, 1, num_neurons_current_layer)
        # self.bias = np.random.rand(num_neurons)
        self.last_layer = last_layer
        self.z = 0  # Result of the layer before the activation function
        self.a = 0  # Result of the layer after the activation function

    def layer_forward_propagation(self, neuron_input):
        z = (self.weights @ neuron_input) + self.bias
        self.z = z

        if self.last_layer:
            # If the current layer is the last one, we do not apply activation function
            self.a = z
        else:
            self.a = ReLU(z)  # "a" in the lecture

        return self.a


class FullyConnectedNeuralNetwork:

    def __init__(self, input_feature_size, num_layer, num_neurons_by_layer, learning_rate):
        assert num_layer == len(num_neurons_by_layer), (f"The size of num_neurons_by_layer : "
                                                        f"{len(num_neurons_by_layer)} should match the num_layer "
                                                        f"{num_layer}")
        self.layers = []
        self.input_feature_size = input_feature_size
        self.num_layer = num_layer
        self.num_neurons_by_layer = num_neurons_by_layer
        self.learning_rate = learning_rate
        self.layers_output = []

    def initialize_layers(self):
        for i in range(self.num_layer):
            # We set the first layer input with the feature input size
            if i == 0:
                self.layers.append(Layer(self.input_feature_size, self.num_neurons_by_layer[i], False))
            # The other layers input size are defined by the previous layers' output size
            elif i == self.num_layer - 1:
                # If it is the last layer, notify it, not to apply the activation function
                self.layers.append(Layer(self.num_neurons_by_layer[i - 1], self.num_neurons_by_layer[i], True))
            else:
                self.layers.append(Layer(self.num_neurons_by_layer[i - 1], self.num_neurons_by_layer[i], False))

    def forward_propagation(self, input_feature):
        # Check that the input array has a valid size
        assert input_feature.ndim == 1, (f"Array does not have the expected 1 dimensions, but has "
                                         f"{input_feature.ndim} dimensions.")
        assert input_feature.shape[0] == self.input_feature_size, (f"Array does not have the expected "
                                                               f"{self.input_feature_size} size, but has "
                                                               f"{input_feature.shape[0]} size.")

        # First compute with the input feature
        previous_layer_output = self.layers[0].layer_forward_propagation(input_feature)
        self.layers_output.insert(0, previous_layer_output)

        # Skip the first layer because it is already computed
        for i, layer in enumerate(self.layers[1:], start=1):
            previous_layer_output = layer.layer_forward_propagation(previous_layer_output)
            self.layers_output.insert(i, previous_layer_output)

        # Return the output of the last layer
        return previous_layer_output

    def backpropagation(self, input_feature, target_output):
        deltas = [0] * self.num_layer

        model_hypothesis = float(self.layers_output[-1][0])
        deltas[-1] = np.array([model_hypothesis - target_output])

        # Calculate deltas for the hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            deltas[i] = np.dot(np.transpose(self.layers[i + 1].weights), deltas[i + 1]) * ReLU_prime(self.layers[i].z)

        # Update weights and biases
        for i in range(len(self.layers)):
            # Compute the gradient for weights and biases
            if i == 0:
                gradient_w = np.outer(deltas[i], input_feature)
            else:
                gradient_w = np.outer(deltas[i], self.layers[i - 1].a)

            gradient_b = deltas[i]

            # Update the weights and biases
            self.layers[i].weights -= self.learning_rate * gradient_w
            self.layers[i].bias -= self.learning_rate * gradient_b

    def train_one_epoch(self, input_feature, target_output):
        self.forward_propagation(input_feature)
        self.backpropagation(input_feature, target_output)


def main():
    input_feature = np.ones(3)
    target_output = 1
    learning_rate = 0.2

    fullyConnectedNN = FullyConnectedNeuralNetwork(3, 3, [3, 2, 1], learning_rate)
    fullyConnectedNN.initialize_layers()
    print("First prediction : ", fullyConnectedNN.forward_propagation(input_feature)[0])
    fullyConnectedNN.train_one_epoch(input_feature, target_output)
    print("Second prediction : ", fullyConnectedNN.forward_propagation(input_feature)[0])


if __name__ == "__main__":
    main()
