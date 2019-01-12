import abc
import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import nn


class _Training(abc.ABC):
    def __init__(self, model: nn.NeuralNetwork, training_data, test_data):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.result = Result(training_data, test_data)

    @abc.abstractmethod
    def train(self):
        pass

    def save_result(self, name: str):
        os.system('mkdir -p models')
        nn.save(self.model, 'models/{}.pkl'.format(name))
        self.result.plot('models/{}.png'.format(name))

    def _remove_saved_models(self):
        os.system('rm ' + self.SAVED_MODEL_NAME_FORMAT.format('*'))


class Training(_Training):
    """Train a model on training data for a number of epochs."""

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 epochs: int):
        super().__init__(model, training_data, test_data)
        self.epochs = epochs

    def train(self):
        """Train the model and collect data."""
        self.result.collect(self.model)
        for _ in range(1, self.epochs + 1):
            self.model.train(self.training_data)
            self.result.collect(self.model)


class EarlyStoppingTraining(_Training):
    """Train a model on the optimal number of epochs."""

    SAVED_MODEL_NAME_FORMAT = '/tmp/validation-epochs-{}.pkl'

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 validation_data:  List[Tuple[np.ndarray, np.ndarray]],
                 max_steps_without_progression=2):
        super().__init__(model, training_data, test_data)
        self.validation_data = validation_data
        self.max_steps_without_progression = max_steps_without_progression

    def train(self):
        """Train a model and collect data."""
        self.result.collect(self.model)
        current_epochs = 0
        optimal_epochs = 0
        current_steps_without_progression = 0
        lowest_loss = sys.float_info.max
        current_loss = lowest_loss

        while current_steps_without_progression < self.max_steps_without_progression:
            print(
                '\n--------------------- Neural Network Training Step ---------------------'
                '    - current epochs {}\n'.format(current_epochs),
                '    - current steps without progression {}\n'.format(
                    current_steps_without_progression),
                '    - current loss {0:1.9f}\n'.format(current_loss))

            self.model.train(self.training_data)
            current_epochs = current_epochs + 1
            current_loss, _ = self.model.test(self.validation_data)

            self.result.collect(self.model)

            if (current_loss < lowest_loss):
                current_steps_without_progression = 0
                self._remove_saved_models()
                nn.save(self.model, self.SAVED_MODEL_NAME_FORMAT.format(current_epochs))
                optimal_epochs = current_epochs
                lowest_loss = current_loss
            else:
                current_steps_without_progression = current_steps_without_progression + 1

        self.model = nn.load(self.SAVED_MODEL_NAME_FORMAT.format(optimal_epochs))
        print('    - optimal epochs {}'.format(optimal_epochs))
        self._remove_saved_models()


class Result():
    """Collect data from model by running test on different data set.

    Examples::
        >>> result = Result(training_data, test_data)
        >>> result.collect(model)
        >>> result.plot('mnist-test-1')
    """

    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        self.training_accuracies = []
        self.test_accuracies = []
        self.training_losses = []
        self.test_losses = []

    def collect(self, model: nn.NeuralNetwork):
        """Collect loss and accuracy on trainind and test data set.

        :param model: The model to be tested
        """
        test_loss, test_accuracy = model.test(self.test_data)
        training_loss, training_accuracy = model.test(self.training_data)

        print(
            '\n------------------ Neural Network Testing Result ------------------\n',
            '   - epoch {0} \n'.format(len(self.test_accuracies)),
            '   - test loss {0:1.9f}\n'.format(test_loss),
            '   - test accuracy {0:1.3f} %\n'.format(test_accuracy),
            '   - training loss {0:1.9f}\n'.format(training_loss),
            '   - training accuracy {0:1.3f} %\n'.format(training_accuracy))

        self.test_accuracies.append(test_accuracy / 100.0)
        self.training_accuracies.append(training_accuracy / 100.0)
        self.test_losses.append(test_loss)
        self.training_losses.append(training_loss)

    def plot(self, file_name: str):
        """Create an graph containing losses and accuracies.

        All data collected with the collect function will be inclused in the graph

        :param file_name: Path that will contain the graph in png format
        """
        self._plot_accuracy()
        self._plot_losses()
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(file_name)
        plt.close()

    def _plot_losses(self):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')

    def _plot_accuracy(self):
        plt.plot(self.training_accuracies, label='Training Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
