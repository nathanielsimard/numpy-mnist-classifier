import numpy as np

SIGMOID = 'sigmoid'
RELU = 'relu'


def sigmoid(x) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x) -> np.ndarray:
    return np.maximum(x, 0)


def relu_derivative(x) -> np.ndarray:
    return 1. * (x > 0)


def get(name: str):
    return _functions[name]


def get_derivative(name: str):
    return _derivatives[name]


_functions = {SIGMOID: sigmoid, RELU: relu}
_derivatives = {SIGMOID: sigmoid_derivative, RELU: relu_derivative}
