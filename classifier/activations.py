import numpy as np

SIGMOID = 'sigmoid'
RELU = 'relu'


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x) -> np.ndarray:
    """Sigmoid derivative function."""
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x) -> np.ndarray:
    """Relu function."""
    return np.maximum(x, 0)


def relu_derivative(x) -> np.ndarray:
    """Relu derivative function."""
    return 1. * (x > 0)


def find(name: str):
    """Find the function from its name.

    :param name: The name of the function in lowercase
    :return: The function
    """
    return _functions[name]


def find_derivative(name: str):
    """Find the derivative function from its name.

    :param name: The name of the function in lowercase
    :return: The derivative function
    """
    return _derivatives[name]


_functions = {SIGMOID: sigmoid, RELU: relu}
_derivatives = {SIGMOID: sigmoid_derivative, RELU: relu_derivative}
