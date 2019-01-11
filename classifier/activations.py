from enum import Enum

import numpy as np


class ActivationFunction(Enum):
    SIGMOID = 'sigmoid'
    RELU = 'relu'

    def get(self):
        return _functions.get(self.name)[0]

    def get_derivative(self):
        return _functions.get(self.name)[1]


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x) -> np.ndarray:
    return np.maximum(x, 0)


def relu_derivative(x) -> np.ndarray:
    return 1. * (x > 0)


_functions = {
    ActivationFunction.SIGMOID.name: (sigmoid, sigmoid_derivative),
    ActivationFunction.RELU.name: (relu, relu_derivative)
}
