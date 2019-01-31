import pickle
from typing import List, Tuple

import numpy as np

from . import activations


class NeuralNetwork():
    """Deep neural network using batch training.

    Example::
        >>> model = NeuralNetwork([784, 30, 10])
        >>> model.train(training_data)
        >>> model.test(test_data)

    .. note::
        To train the model for a number of epochs you should call
        ``model.train(training_data)`` multiple times
    """

    def __init__(self,
                 layers_size: List[int],
                 learning_rate=0.01,
                 batch_size=10,
                 activation=activations.SIGMOID,
                 weights=None,
                 biaises=None):
        self.learning_rate = learning_rate
        self.layers_size = layers_size
        self.number_layers = len(layers_size) - 1
        self.batch_size = batch_size
        self.activation = activation
        self._generate_weights = _generate_random
        self._activation = activations.find(activation)
        self._activation_derivative = activations.find_derivative(activation)
        self.weights = weights
        self.biaises = biaises

        if (self.weights is None):
            self._initialize_weights(layers_size)
        if (self.biaises is None):
            self._initialize_biases(layers_size)

    def train(self, data_set: List[Tuple[np.ndarray, np.ndarray]]):
        """Train the neural network on input data and adjust it weights ans biaises.

        :param data_set: data used to train the neural network
        """
        batch_gradients_w = self._initialize_batch(self.weights)
        batch_gradients_b = self._initialize_batch(self.biaises)

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
                batch_gradients_b = self._initialize_batch(self.biaises)

    def test(self, data_set: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[float, float]:
        """Test the neural network on the input data set.

        :param data_set: The data that will be used to test the model
        :return: Losses and accuracies
        """
        misses = 0
        total = 0
        losses = 0.0
        for x, y in zip(data_set[0], data_set[1]):
            total += 1
            _, a = self.forward_propagation(x)
            losses += self._loss(a, y)
            prediction = np.argmax(a[self.number_layers])
            expected = np.argmax(y)
            if (prediction != expected):
                misses += 1
        losses = losses / total
        accuracy = 100 * (total - misses) / total

        return losses, accuracy

    def forward_propagation(self, data: np.ndarray):
        """Forward_propagation algorithms.

        :param data: A numpy array containing input data such as an image
        :return: (z, a) Each layer and activation
        """
        z = _create_array(self.number_layers + 1)
        a = _create_array(self.number_layers + 1)
        a[0] = data
        for k in range(1, self.number_layers + 1):
            z[k] = np.dot(self.weights[k], a[k - 1]) + self.biaises[k]
            a[k] = self._activation(z[k])
        return z, a

    def backward_propagation(self,
                             z: List[np.ndarray],
                             a: List[np.ndarray],
                             y: np.ndarray):
        """Backward_propagation algorothm.

        :param z: An array containing each layer
        :param a: An array containing activations of each layer
        :param y: A numpy array containing data label
        :return: (gradients_w, gradients_b) Computed weights and biaises gradients
        """
        gradients = _create_array(self.number_layers + 1)
        cost_derivative = self._error_derivative(a, y)
        propagation = cost_derivative

        for k in range(self.number_layers, 0, -1):
            a_derivative = self._activation_derivative(z[k])
            gradients[k] = propagation * a_derivative
            propagation = np.dot(self.weights[k].T, gradients[k])

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
        self.biaises = [None]  # B_0 does not exist
        for i in range(1, len(layers_size)):
            layer = layers_size[i]
            self.biaises.append(self._generate_weights((layer, 1)))

    def _loss(self, a, y):
        cost = self._error(a, y)
        return np.sum(cost)

    def _error(self, a, y):
        prediction = a[self.number_layers]
        error = (prediction - y)
        return np.square(error)

    def _error_derivative(self, a, y):
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
            self.biaises[i + 1] -= self.learning_rate * gradients_b[i]


def save(model: NeuralNetwork, file_name: str):
    """Save the input model using pickle.

    :param model: The neural network to be saved
    :param file_name: The path to pickle file
    """
    parameters = {
        'learning_rate': model.learning_rate,
        'layers_size': model.layers_size,
        'number_layers': model.number_layers,
        'batch_size': model.batch_size,
        'activation_function': model.activation,
        'weights': model.weights,
        'biases': model.biaises
    }
    with open(file_name, 'wb') as file:
        pickle.dump(parameters, file)


def load(file_name: str) -> NeuralNetwork:
    """Load a neural network model.

    :param file_name: The pickle file containing a saved neural networl
    :return: The neural network already initialized or trained
    """
    with open(file_name, 'rb') as file:
        parameters = pickle.load(file)
        return NeuralNetwork(
            parameters['layers_size'],
            learning_rate=parameters['learning_rate'],
            batch_size=parameters['batch_size'],
            activation=parameters['activation_function'],
            weights=parameters['weights'],
            biaises=parameters['biases'])


def _create_array(size):
    return [None] * (size)


def _generate_random(shape) -> np.ndarray:
    return 2 * np.random.random(shape) - 1
