import numpy as np


"""
III) Deep learning
Chapter 7
Deep learning
"""

"""
Neural networks
"""


"""
This implementation with a Neuron class do not seems relevant. It doesn't allow to use a matrix of weights in the Layer 
class. To speed up the calculus, it is much more convenient to use the a matrix of weights. Instead here, I use a for
lop to trigger the calculus in each neuron.
"""

np.random.seed(42)  # For reproducibility


class Neuron:
    def __init__(self, num_neurons_previous_layer, last_layer):
        self.weights = np.random.rand(1, num_neurons_previous_layer)
        self.bias = np.random.uniform(-1, 1, 1)
        self.last_layer = last_layer
        self.num_neurons_previous_layer = num_neurons_previous_layer

    def neuron_forward_propagation(self, neuron_input):
        z = (self.weights @ neuron_input) + self.bias
        print("WEIGHTS : ", self.weights)
        print("BIAS : ", self.bias)
        print("=========================== Z : ", z[0])
        if self.last_layer:
            print("=========================== A : ", z[0])
            # Using float() to replace np.float (ugly display)
            return float(z[0])  # If the current layer is the last one, we do not apply activation function
        else:
            print("=========================== A : ", self.ReLU(z[0]))
            # Using float() to replace np.float (ugly display)
            return float(self.ReLU(z[0]))  # a in the lecture

    def ReLU(self, z):
        return np.maximum(z, 0)


class Layer:
    def __init__(self, num_neurons_previous_layer, num_neurons, last_layer):
        self.neurons = []
        self.num_neurons_previous_layer = num_neurons_previous_layer
        self.num_neurons = num_neurons
        self.last_layer = last_layer
        self.initialize_neurons()

    def initialize_neurons(self):
        for i in range(self.num_neurons):
            self.neurons.append(Neuron(self.num_neurons_previous_layer, self.last_layer))

    def layer_forward_propagation(self, neuron_input):
        layer_output = []

        for neuron in self.neurons:
            layer_output.append(neuron.neuron_forward_propagation(neuron_input))

        print("layer ouput : ", layer_output)
        return layer_output


class FullyConnectedNeuralNetwork:

    def __init__(self, input_feature_size, num_layer, num_neurons_by_layer):
        assert num_layer == len(num_neurons_by_layer), (f"The size of num_neurons_by_layer : "
                                                        f"{len(num_neurons_by_layer)} should match the num_layer "
                                                        f"{num_layer}")
        self.layers = []
        self.input_feature_size = input_feature_size
        self.num_layer = num_layer
        self.num_neurons_by_layer = num_neurons_by_layer
        self.initialize_layers()

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
    print(fullyConnectedNN.forward_propagation(input_feature))


if __name__ == "__main__":
    main()
