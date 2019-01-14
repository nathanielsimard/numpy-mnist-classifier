import abc
import os
import shutil
import sys
from typing import List, Tuple

import numpy as np

from . import nn
from .result import Result, ResultCollector


class _Training(abc.ABC):
    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data,
                 validation_data,
                 test_data):
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.result_collector = ResultCollector()

    @abc.abstractmethod
    def train(self) -> Result:
        pass

    def _result(self, trained_model: nn.NeuralNetwork, training_method: str) -> Result:
        return Result(self.training_data,
                      self.test_data,
                      self.validation_data,
                      self.result_collector.training_accuracies,
                      self.result_collector.test_accuracies,
                      self.result_collector.training_losses,
                      self.result_collector.test_losses,
                      self.result_collector.epochs,
                      trained_model,
                      training_method)

    def collect_stat(self, epoch):
        training_loss, training_accuracy = self.model.test(self.training_data)
        test_loss, test_accuracy = self.model.test(self.test_data)
        self.result_collector.collect(epoch, test_loss, test_accuracy, training_loss, training_accuracy)


class Basic(_Training):

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 validation_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 epochs: int):
        super().__init__(model, training_data, validation_data, test_data)
        self.epochs = epochs

    def train(self) -> Result:
        """Train the model and collect data."""
        self.collect_stat(0)
        for epoch in range(1, self.epochs + 1):
            self.model.train(self.training_data)
            self.collect_stat(epoch)
        return self._result(self.model, 'basic')


class EarlyStoppingRegularization(_Training):
    SAVED_MODEL_FOLDER = '.tmp'
    SAVED_MODEL_FILE_FORMAT = '/validation-epochs-{}.pkl'

    def __init__(self,
                 model: nn.NeuralNetwork,
                 training_data:  List[Tuple[np.ndarray, np.ndarray]],
                 test_data:  List[Tuple[np.ndarray, np.ndarray]],
                 validation_data:  List[Tuple[np.ndarray, np.ndarray]],
                 epochs=1,
                 max_steps_without_progression=2):
        super().__init__(model, training_data, validation_data, test_data)
        self.epochs = epochs
        self.max_steps_without_progression = max_steps_without_progression

    def train(self) -> Result:
        """Train the network until there is no more improvement on the validation data."""
        current_epoch = 0
        optimal_epoch = 0
        current_steps_without_progression = 0
        lowest_loss = sys.float_info.max
        current_loss = lowest_loss

        self.collect_stat(current_epoch)
        os.makedirs(self.SAVED_MODEL_FOLDER, exist_ok=True)
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
                nn.save(self.model, self.SAVED_MODEL_FOLDER + self.SAVED_MODEL_FILE_FORMAT.format(current_epoch))
                optimal_epoch = current_epoch
                lowest_loss = current_loss
            else:
                current_steps_without_progression = current_steps_without_progression + 1

        print('    - optimal epochs {}'.format(optimal_epoch))
        trained_model = nn.load(self.SAVED_MODEL_FOLDER + self.SAVED_MODEL_FILE_FORMAT.format(optimal_epoch))
        self._remove_saved_models()
        return self._result(trained_model, 'early stopping regularization')

    def _remove_saved_models(self):
        shutil.rmtree(self.SAVED_MODEL_FOLDER)
        os.makedirs(self.SAVED_MODEL_FOLDER)
