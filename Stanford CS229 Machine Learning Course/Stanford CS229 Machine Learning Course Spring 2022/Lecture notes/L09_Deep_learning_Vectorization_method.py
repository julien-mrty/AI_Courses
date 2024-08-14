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
Here I use what I did previously, add the backpropagation and improve my code
Note : I use the NN for logistic regression to simplify the samples creation. By choice, I do not apply activation 
function on my output layer
"""

#np.random.seed(42)  # For reproducibility


def ReLU(z):
    return np.maximum(z, 0)


def ReLU_prime(z):
    return np.where(z < 0, 0, 1)


def sigmoid(z):
    z = np.clip(z, -500, 500)  # Clip to avoid overflow
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    return np.tanh(z)


def tanh_prime(z):
    return 1 - np.tanh(z) ** 2


class ActivationFunctions:
    functions = {
        'relu': (ReLU, ReLU_prime),
        'sigmoid': (sigmoid, sigmoid_prime),
        'tanh': (tanh, tanh_prime)
    }

    @staticmethod
    def get(name):
        return ActivationFunctions.functions.get(name, (None, None))


class FullyConnectedLayer:
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer, act_func=None):
        self.weights = np.random.rand(num_neurons_current_layer, num_neurons_previous_layer)
        self.bias = np.zeros((num_neurons_current_layer, 1))
        self.z = None  # Output of the layer before the activation function
        self.a = None  # Output of the layer after the activation function
        self.activation_function = act_func

    def initialize_weights(self, num_neurons_previous_layer, num_neurons_current_layer):
        # He initialization for ReLU, Xavier initialization for others
        if self.activation_function == ReLU:
            stddev = np.sqrt(2. / num_neurons_previous_layer)
            return np.random.randn(num_neurons_current_layer, num_neurons_previous_layer) * stddev
        else:
            stddev = np.sqrt(1. / num_neurons_previous_layer)
            return np.random.randn(num_neurons_current_layer, num_neurons_previous_layer) * stddev

    def layer_forward_propagation(self, neuron_input):
        self.z = np.dot(self.weights, neuron_input) + self.bias
        # Apply activation function only if it's provided
        self.a = self.activation_function(self.z) if self.activation_function else self.z
        return self.a


class OutputLayer(FullyConnectedLayer):
    def __init__(self, num_neurons_previous_layer, num_neurons_current_layer):
        super().__init__(num_neurons_previous_layer, num_neurons_current_layer)


class LayerFactory:
    @staticmethod
    def create_layer(layer_type, num_neurons_previous_layer, num_neurons_current_layer, activation_function=None):
        if layer_type == 'output':
            return OutputLayer(num_neurons_previous_layer, num_neurons_current_layer)
        else:
            return FullyConnectedLayer(num_neurons_previous_layer, num_neurons_current_layer, activation_function)


class FullyConnectedNeuralNetwork:
    def __init__(self, input_feature_size, num_neurons_by_layer, learning_rate, act_func="relu"):
        # Hyperparameters
        self.learning_rate = learning_rate

        # Layers
        self.input_feature_size = input_feature_size
        self.layers = []
        self.layers_output = []
        self.num_neurons_by_layer = num_neurons_by_layer
        self.num_layer = len(self.num_neurons_by_layer)

        # Get activation functions
        self.activation_function, self.activation_function_prime = ActivationFunctions.get(act_func)

        # Initialization of NN layers
        self.initialize_layers()

    def initialize_layers(self):
        for i in range(self.num_layer):
            input_size = self.input_feature_size if i == 0 else self.num_neurons_by_layer[i - 1]
            layer_type = 'output' if i == self.num_layer - 1 else 'hidden'
            layer = LayerFactory.create_layer(layer_type, input_size, self.num_neurons_by_layer[i], self.activation_function if layer_type == 'hidden' else None)
            self.layers.append(layer)

    def set_activation_function(self, act_func, act_func_prime):
        self.activation_function = act_func
        self.activation_function_prime = act_func_prime

        for layer in self.layers:
            layer.activation_function = self.activation_function

    def forward_propagation(self, input_feature):
        previous_layer_output = input_feature.T
        for layer in self.layers:
            previous_layer_output = layer.layer_forward_propagation(previous_layer_output)

        # Return the output of the last layer
        return previous_layer_output

    def backpropagation(self, input_feature, target_output):
        n = input_feature.shape[0]  # Number of samples

        # Initialize delta
        deltas = [None] * self.num_layer

        # Last layer delta
        last_layer = self.layers[-1]
        deltas[-1] = last_layer.a - target_output.T  # (1, m)

        # Back-propagate through layers
        # First compute the deltas
        for i in range(self.num_layer - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            deltas[i] = np.dot(next_layer.weights.T, deltas[i + 1]) * self.activation_function_prime(layer.z)

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_activation = input_feature.T  # (input_size, m)
            else:
                input_activation = self.layers[i - 1].a  # (neurons_in_previous_layer, m)

            gradient_weight = np.dot(deltas[i], input_activation.T) / n
            gradient_bias = np.sum(deltas[i], axis=1, keepdims=True) / n

            # Update the weights and biases
            layer.weights -= self.learning_rate * gradient_weight
            layer.bias -= self.learning_rate * gradient_bias

    def train_one_epoch(self, batch_size, input_feature, target_output):
        n = input_feature.shape[0]
        if n != target_output.shape[0]:
            raise ValueError(f"Input and target do not match: {n} vs {target_output.shape[0]}.")

        for i in range(0, n, batch_size):
            # Slice the batches
            batch_input = input_feature[i:i + batch_size, :]
            batch_target = target_output[i:i + batch_size]

            # Perform forward propagation and backpropagation on the current batch
            self.forward_propagation(batch_input)
            self.backpropagation(batch_input, batch_target)

    def train(self, number_of_epochs, batch_size, input_feature, target_output):
        for _ in range(number_of_epochs):
            self.train_one_epoch(batch_size, input_feature, target_output)


def main():
    """ Hyperparameters """
    learning_rate = 0.01
    number_of_epochs = 1
    batch_size = 8

    """ Samples """
    input_feature_size = 4
    number_of_samples = 100
    number_of_samples_per_class = number_of_samples // 2

    input_feature_0 = np.random.normal(loc=-1, scale=0.5, size=(number_of_samples_per_class, input_feature_size))
    input_feature_1 = np.random.normal(loc=1, scale=0.5, size=(number_of_samples_per_class, input_feature_size))
    input_feature = np.concatenate((input_feature_0, input_feature_1), axis=0)

    target_output_0 = np.zeros(number_of_samples_per_class)
    target_output_1 = np.ones(number_of_samples_per_class)
    target_output = np.concatenate((target_output_0, target_output_1), axis=0)

    """ Neural network """
    # Create the NN
    fullyConnectedNN = FullyConnectedNeuralNetwork(input_feature_size, [2, 1], learning_rate, "sigmoid")

    # Print input and expected values (I test the two classes)
    print("Input 1 : ", input_feature[0], ", expected result : ", target_output[0])
    print("Input 2 : ", input_feature[50], ", expected result : ", target_output[50])

    # Print result before training (I test the two classes)
    initial_predictions = fullyConnectedNN.forward_propagation(input_feature)
    print("Input 1 initial prediction : ", initial_predictions[0, 0])
    print("Input 2 initial prediction : ", initial_predictions[0, 50])

    # Train the NN
    fullyConnectedNN.train(number_of_epochs, batch_size, input_feature, target_output)

    # Print result after training (I test the two classes)
    final_predictions = fullyConnectedNN.forward_propagation(input_feature)
    print("Input 1 final prediction : ", final_predictions[0, 0])
    print("Input 2 final prediction : ", final_predictions[0, 50])


if __name__ == "__main__":
    main()
