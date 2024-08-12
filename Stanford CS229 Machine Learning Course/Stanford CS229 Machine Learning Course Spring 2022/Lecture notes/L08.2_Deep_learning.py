import numpy as np


"""
III) Deep learning
Chapter 7
Deep learning
"""

np.random.seed(42)  # For reproducibility


class Layer:
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, last_layer):
        self.weights = np.random.rand(num_neurons_current_layer, num_neurons_previous_layer)
        # self.weights = np.random.uniform(-1, 1, (num_neurons, num_neurons_previous_layer))
        self.bias = np.random.uniform(-1, 1, num_neurons_current_layer)
        # self.bias = np.random.rand(num_neurons)
        self.last_layer = last_layer

    def layer_forward_propagation(self, neuron_input):
        z = (self.weights @ neuron_input) + self.bias
        print("WEIGHTS : ", self.weights)
        print("BIAS : ", self.bias)
        print("=========================== Z : ", z)
        if self.last_layer:
            print("=========================== A : ", z)
            return z  # If the current layer is the last one, we do not apply activation function
        else:
            print("=========================== A : ", self.ReLU(z))
            return self.ReLU(z)  # a in the lecture

    def ReLU(self, z):
        return np.maximum(z, 0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


class FullyConnectedNeuralNetwork:

    def __init__(self, input_feature_size, num_layer, num_neurons_by_layer):
        assert num_layer == len(num_neurons_by_layer), (f"The size of num_neurons_by_layer : "
                                                        f"{len(num_neurons_by_layer)} should match the num_layer "
                                                        f"{num_layer}")
        self.layers = []
        self.input_feature_size = input_feature_size
        self.num_layer = num_layer
        self.num_neurons_by_layer = num_neurons_by_layer

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

        # Skip the first layer because it is already computed
        for layer in self.layers[1:]:
            previous_layer_output = layer.layer_forward_propagation(previous_layer_output)

        # Return the output of the last layer
        return previous_layer_output


def main():
    input_feature = np.ones(3)
    fullyConnectedNN = FullyConnectedNeuralNetwork(3, 3, [3, 2, 1])
    fullyConnectedNN.initialize_layers()
    print(fullyConnectedNN.forward_propagation(input_feature))


if __name__ == "__main__":
    main()
