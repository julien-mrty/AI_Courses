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

#np.random.seed(42)  # For reproducibility


def ReLU(z):
    return np.maximum(z, 0)


def ReLU_prime(z):
    return np.where(z < 0, 0, 1)  # Return 1 if z > 0 or 0


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return np.exp(-z) / ((1 + np.exp(-z)) ** 2)


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


class Layer:
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, last_layer, act_func):
        self.weights = np.random.rand(num_neurons_current_layer, num_neurons_previous_layer)
        self.bias = np.random.uniform(-0.5, 0.5, num_neurons_current_layer)
        self.last_layer = last_layer
        self.z = 0  # Result of the layer before the activation function
        self.a = 0  # Result of the layer after the activation function
        self.activation_function = act_func

    def set_activation_function(self, act_func):
        self.activation_function = act_func

    def layer_forward_propagation(self, neuron_input):
        z = (self.weights @ neuron_input) + self.bias
        self.z = z

        if self.last_layer:
            # If the current layer is the last one, we do not apply activation function
            self.a = self.z
        else:
            self.a = self.activation_function(self.z)

        return self.a


class FullyConnectedNeuralNetwork:

    def __init__(self, input_feature_size, num_neurons_by_layer, learning_rate):
        # Hyperparameters
        self.learning_rate = learning_rate

        # Layers
        self.input_feature_size = input_feature_size
        self.layers = []
        self.layers_output = []
        self.num_neurons_by_layer = num_neurons_by_layer
        self.num_layer = len(self.num_neurons_by_layer)

        # Activation function
        self.activation_function = ReLU
        self.activation_function_prime = ReLU_prime

        # Initialization of NN layers
        self.initialize_layers()

    def initialize_layers(self):
        for i in range(self.num_layer):
            # We set the first layer input with the feature input size
            if i == 0:
                if i == self.num_layer - 1:
                    self.layers.append(Layer(self.input_feature_size, self.num_neurons_by_layer[i], True,
                                             self.activation_function))
                else:
                    self.layers.append(Layer(self.input_feature_size, self.num_neurons_by_layer[i], False,
                                             self.activation_function))
            # The other layers input size are defined by the previous layers' output size
            elif i == self.num_layer - 1:
                # If it is the last layer, notify it, not to apply the activation function
                self.layers.append(Layer(self.num_neurons_by_layer[i - 1], self.num_neurons_by_layer[i], True,
                                         self.activation_function))
            else:
                self.layers.append(Layer(self.num_neurons_by_layer[i - 1], self.num_neurons_by_layer[i], False,
                                         self.activation_function))

    def set_activation_function(self, act_func, act_func_prime):
        self.activation_function = act_func
        self.activation_function_prime = act_func_prime

        for layer in self.layers:
            layer.set_activation_function(self.activation_function)

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
        # Initialization of the deltas' list
        deltas = [0] * self.num_layer

        # Get the model hypothesis
        model_hypothesis = float(self.layers_output[-1][0])
        deltas[-1] = np.array([model_hypothesis - target_output])

        # Calculate deltas for the hidden layers
        for i in range(len(self.layers) - 2, -1, -1):
            deltas[i] = (np.dot(np.transpose(self.layers[i + 1].weights), deltas[i + 1]) *
                         self.activation_function_prime(self.layers[i].z))

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

    def train_one_example(self, input_feature, target_output):
        self.forward_propagation(input_feature)
        self.backpropagation(input_feature, target_output)

    def train_one_epoch(self, input_feature, target_output):
        assert input_feature.shape[0] == target_output.shape[0], (f"The input feature and the target output doesn't "
                                                                  f"contain the same number of samples : input feature "
                                                                  f"{input_feature.shape[0]},"
                                                                  f"target output {target_output.shape[0]}.")

        for i in range(target_output.shape[0]):
            self.forward_propagation(input_feature[i])
            self.backpropagation(input_feature[i], target_output[i])

    def train(self, number_of_epochs, input_feature, target_output):
        for i in range(number_of_epochs):
            self.train_one_epoch(input_feature, target_output)


def main():
    """ Hyperparameters """
    learning_rate = 0.001
    number_of_epochs = 1000


    """ Samples """
    input_feature_size = 4
    number_of_samples = 100
    number_of_samples_per_class = int(number_of_samples / 2)

    input_feature_0 = np.random.normal(loc=-1, scale=0.5, size=(number_of_samples_per_class, input_feature_size))
    input_feature_1 = np.random.normal(loc=1, scale=0.5, size=(number_of_samples_per_class, input_feature_size))
    input_feature = np.concatenate((input_feature_0, input_feature_1), axis=0)

    target_output_0 = np.zeros(number_of_samples_per_class)
    target_output_1 = np.ones(number_of_samples_per_class)
    target_output = np.concatenate((target_output_0, target_output_1), axis=0)


    """ Neural network """
    fullyConnectedNN = FullyConnectedNeuralNetwork(input_feature_size, [2, 1], learning_rate)
    fullyConnectedNN.set_activation_function(sigmoid, sigmoid_prime)

    print("Input 1 : ", input_feature[0], ", expected result : ", target_output[0])
    print("Input 2 : ", input_feature[50], ", expected result : ", target_output[50])
    print("Input 1 first prediction : ", fullyConnectedNN.forward_propagation(input_feature[0])[0])
    print("Input 2 first prediction : ", fullyConnectedNN.forward_propagation(input_feature[50])[0])

    fullyConnectedNN.train(number_of_epochs, input_feature, target_output)

    print("Input 1 second prediction : ", fullyConnectedNN.forward_propagation(input_feature[0])[0])
    print("Input 2 second prediction : ", fullyConnectedNN.forward_propagation(input_feature[50])[0])


if __name__ == "__main__":
    main()
