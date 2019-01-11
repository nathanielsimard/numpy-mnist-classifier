import os
import sys

import numpy as np

from data import mnist
from nn import NeuralNetwork, save
from regularization import EarlyStoppingRegularization
from result import Result


def train(regularization=True):
    training_data, validation_data, test_data = mnist.load()
    model = NeuralNetwork(
        [784, 30, 10],
        batch_size=50,
        learning_rate=0.02,
    )
    result = Result(training_data, test_data)

    if regularization:
        early_stopping_regularization = EarlyStoppingRegularization(
            training_data,
            validation_data,
            result,
            max_steps_without_progression=2)
        model = early_stopping_regularization.train(model)
    else:
        result.collect(model)
        for _ in range(1, 2):
            model.train(training_data)
            result.collect(model)

    os.system('mkdir -p models')
    save(model, 'models/mnist-model.pkl')
    result.plot('models/mnist-model-result.png')


if __name__ == "__main__":
    train()
