import abc
import os
import sys
from typing import List, Tuple

import numpy as np
from .result import Result, ResultCollector

from . import nn


def train(model,
          training_data,
          test_data,
          epochs=1,
          validation_data=([], []),
          early_stopping_regularization=False) -> Result:
    """Train the model and return the result."""
    training = _create_training(model,
                                training_data,
                                test_data,
                                validation_data,
                                epochs,
                                early_stopping_regularization)
    trained_model = training.train()
    return Result(training_data,
                  test_data,
                  validation_data,
                  training.result_collector.training_accuracies,
                  training.result_collector.test_accuracies,
                  training.result_collector.training_losses,
                  training.result_collector.test_losses,
                  training.result_collector.epochs,
                  trained_model)


class _Training(abc.ABC):
    def __init__(self, model: nn.NeuralNetwork, training_data, test_data):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data
        self.result_collector = ResultCollector()

    @abc.abstractmethod
    def train(self) -> nn.NeuralNetwork:
        pass

    def collect_stat(self, epoch):
        training_loss, training_accuracy = self.model.test(self.training_data)
        test_loss, test_accuracy = self.model.test(self.test_data)
        self.result_collector.collect(epoch, test_loss, test_accuracy, training_loss, training_accuracy)

    def _remove_saved_models(self):
        os.system('rm ' + self.SAVED_MODEL_NAME_FORMAT.format('*'))


class _BasicTraining(_Training):

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 epochs: int):
        super().__init__(model, training_data, test_data)
        self.epochs = epochs

    def train(self) -> nn.NeuralNetwork:
        """Train the model and collect data."""
        self.collect_stat(0)
        for epoch in range(1, self.epochs + 1):
            self.model.train(self.training_data)
            self.collect_stat(epoch)
        return self.model


class _EarlyStowGppingTraining(_Training):
    SAVED_MODEL_NAME_FORMAT = '/tmp/validation-epochs-{}.pkl'

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 validation_data:  List[Tuple[np.ndarray, np.ndarray]],
                 epochs,
                 max_steps_without_progression=2):
        super().__init__(model, training_data, test_data)
        self.validation_data = validation_data
        self.epochs = epochs
        self.max_steps_without_progression = max_steps_without_progression

    def train(self) -> nn.NeuralNetwork:
        current_epoch = 0
        optimal_epoch = 0
        current_steps_without_progression = 0
        lowest_loss = sys.float_info.max
        current_loss = lowest_loss

        self.collect_stat(current_epoch)
        while current_steps_without_progression < self.max_steps_without_progression:
            print(
                '\n--------------------- Neural Network Training Step ---------------------'
                '    - current epochs {}\n'.format(current_epoch),
                '    - current steps without progression {}\n'.format(
                    current_steps_without_progression),
                '    - current loss {0:1.9f}\n'.format(current_loss))

            for _ in range(0, self.epochs):
                self.model.train(self.training_data)
                current_epoch = current_epoch + 1
                self.collect_stat(current_epoch)

            current_loss, _ = self.model.test(self.validation_data)
            if (current_loss < lowest_loss):
                current_steps_without_progression = 0
                self._remove_saved_models()
                nn.save(self.model, self.SAVED_MODEL_NAME_FORMAT.format(current_epoch))
                optimal_epoch = current_epoch
                lowest_loss = current_loss
            else:
                current_steps_without_progression = current_steps_without_progression + 1

        print('    - optimal epochs {}'.format(optimal_epoch))
        model = nn.load(self.SAVED_MODEL_NAME_FORMAT.format(optimal_epoch))
        self._remove_saved_models()
        return model


def _create_training(model,
                     training_data,
                     test_data,
                     validation_data,
                     epochs,
                     early_stopping_regularization) -> _Training:
    if early_stopping_regularization:

        if len(validation_data[0]) == 0:
            validation_data = test_data

        return _EarlyStowGppingTraining(model,
                                        training_data,
                                        test_data,
                                        validation_data,
                                        epochs)
    else:
        return _BasicTraining(model,
                              training_data,
                              test_data,
                              epochs)
