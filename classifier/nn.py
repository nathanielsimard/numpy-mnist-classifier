import pickle
from typing import List

import numpy as np

from activations import ActivationFunction


class NeuralNetwork():
    def __init__(self,
                 layers_size: int,
                 learning_rate=0.01,
                 batch_size=10,
                 activation=ActivationFunction.SIGMOID,
                 weights=None,
                 biaises=None):
        self.learning_rate = learning_rate
        self.layers_size = layers_size
        self.number_layers = len(layers_size) - 1
        self.batch_size = batch_size
        self.activation = activation
        self._generate_weights = _generate_random
        self._activation = activation.get()
        self._activation_derivative = activation.get_derivative()
        self.weights = weights
        self.biases = biaises

        if (self.weights == None):
            self._initialize_weights(layers_size)
        if (self.biases == None):
            self._initialize_biases(layers_size)

    def train(self, data_set):
        """Train the neural network using multiple batches.

        Args:
            data_set: An array containing labeled data
        """
        batch_gradients_w = self._initialize_batch(self.weights)
        batch_gradients_b = self._initialize_batch(self.biases)

        X = data_set[0]
        Y = data_set[1]
        N = len(X)

        for iteration in range(1, N + 1):
            x = X[iteration - 1]
            y = Y[iteration - 1]

            z, a = self.forward_propagation(x)
            gradients_w, gradients_b = self.backward_propagation(z, a, y)

            self._update_batch(batch_gradients_w, gradients_w)
            self._update_batch(batch_gradients_b, gradients_b)

            if (iteration % self.batch_size == 0):
                self._apply_gradients(batch_gradients_w, batch_gradients_b)

                batch_gradients_w = self._initialize_batch(self.weights)
                batch_gradients_b = self._initialize_batch(self.biases)

    def test(self, data_set, print_result=False):
        """Test the neural network on the input data set.

        Args:
            data_set: An array containing labeled data
        Returns: 
            losses: Sum of losses for each prediction
            accuracy: Missing predictions in percentage
        """
        misses = 0
        total = 0
        losses = 0.0
        for x, y in zip(data_set[0], data_set[1]):
            total += 1
            _, a = self.forward_propagation(x)
            losses += self._lost(a, y)
            prediction = np.argmax(a[self.number_layers])
            expected = np.argmax(y)
            if (prediction != expected):
                misses += 1
        losses = losses / total
        accuracy = 100 * (total - misses) / total

        return losses, accuracy

    def forward_propagation(self, input_data: np.ndarray):
        """Execute the foward propagation algorithm.

        Args:
            input_data: A numpy array containing input data such as an image
        Returns: 
            z: An array containing each layers
            a: An array containing activations of each layer
        """
        z = _create_array(self.number_layers + 1)
        a = _create_array(self.number_layers + 1)
        a[0] = input_data
        for k in range(1, self.number_layers + 1):
            z[k] = np.dot(self.weights[k], a[k - 1]) + self.biases[k]
            a[k] = self._activation(z[k])
        return z, a

    def backward_propagation(self, z: List[np.ndarray], a: List[np.ndarray],
                             y: np.ndarray):
        """Execute the backward propagation algorithm.

        Args:
            z: An array containing each layers
            a: An array containing activations of each layer
            y: A numpy array containing input data label
        Returns: 
            gradients_w: An array containing weights gradients for each layers
            gradients_b: An array containing biases gradients for each layers
        """
        gradients = _create_array(self.number_layers + 1)
        cost_derivative = self._cost_derivative(a, y)
        delta = cost_derivative

        for k in range(self.number_layers, 0, -1):
            a_derivative = self._activation_derivative(z[k])
            gradients[k] = delta * a_derivative
            delta = np.dot(self.weights[k].T, gradients[k])

        gradients_b = _create_array(self.number_layers)
        gradients_w = _create_array(self.number_layers)
        for k in range(self.number_layers, 0, -1):
            gradients_b[k - 1] = gradients[k]
            gradients_w[k - 1] = np.dot(gradients[k], a[k - 1].T)

        return gradients_w, gradients_b

    def _initialize_batch(self, numpy_arrays: List[np.ndarray]):
        batch_gradients = []
        for i in range(1, len(numpy_arrays)):
            batch_gradients.append(np.zeros(numpy_arrays[i].shape))
        return batch_gradients

    def _initialize_weights(self, layers_size):
        preview = layers_size[0]
        self.weights = [None]  # W_0 does not exist
        for i in range(1, len(layers_size)):
            layer = layers_size[i]
            self.weights.append(self._generate_weights((layer, preview)))
            preview = layer

    def _initialize_biases(self, layers_size):
        self.biases = [None]  # B_0 does not exist
        for i in range(1, len(layers_size)):
            layer = layers_size[i]
            self.biases.append(self._generate_weights((layer, 1)))

    def _lost(self, a, y):
        cost = self._cost(a, y)
        return np.sum(cost)

    def _cost(self, a, y):
        prediction = a[self.number_layers]
        error = (prediction - y)
        return np.square(error)

    def _cost_derivative(self, a, y):
        prediction = a[self.number_layers]
        error = (prediction - y)
        return 2 * error

    def _update_batch(self, batch_gradients: List[np.ndarray],
                      gradients: List[np.ndarray]):
        for batch_gradient, gradient in zip(batch_gradients, gradients):
            batch_gradient += gradient

    def _apply_gradients(self, gradients_w: np.ndarray,
                         gradients_b: np.ndarray):
        for i in range(self.number_layers):
            self.weights[i + 1] -= self.learning_rate * gradients_w[i]
            self.biases[i + 1] -= self.learning_rate * gradients_b[i]


def save(model: NeuralNetwork, file_name: str):
    """Save the input model in pkl format

    Args:
        model: The neural network to save
        file_name: Name of the file
    """
    parameters = {
        'learning_rate': model.learning_rate,
        'layers_size': model.layers_size,
        'number_layers': model.number_layers,
        'batch_size': model.batch_size,
        'activation_function': model.activation,
        'weights': model.weights,
        'biases': model.biases
    }
    file = open(file_name, 'wb')
    pickle.dump(parameters, file)
    file.close()


def load(file_name: str) -> NeuralNetwork:
    """Load a neural network model.

    Args:
        file_name: Name of the file
    Returns:
        model: A NeuralNetwork already initiazed or trained
    """
    file = open(file_name, 'rb')
    parameters = pickle.load(file)
    file.close()
    model = NeuralNetwork(
        parameters['layers_size'],
        learning_rate=parameters['learning_rate'],
        batch_size=parameters['batch_size'],
        activation=parameters['activation_function'],
        weights=parameters['weights'],
        biaises=parameters['biases'])
    return model


def _create_array(size):
    """Create an array of the input size"""
    return [None] * (size)


def _generate_random(shape) -> np.ndarray:
    """ Generate random value between -1 and 1 with a mean of 0 """
    return 2 * np.random.random(shape) - 1
