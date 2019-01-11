import os
import sys

from nn import NeuralNetwork, load, save
from result import Result

SAVED_MODEL_NAME_FORMAT = '/tmp/validation-epochs-{}.pkl'


class EarlyStoppingRegularization():
    def __init__(self,
                 training_data,
                 validation_data,
                 result: Result,
                 max_steps_without_progression=2):
        self.training_data = training_data
        self.validation_data = validation_data
        self.result = result
        self.max_steps_without_progression = max_steps_without_progression

    def train(self, model: NeuralNetwork) -> NeuralNetwork:
        self.result.collect(model)
        current_epochs = 0
        optimal_epochs = 0
        current_steps_without_progression = 0
        lowest_loss = sys.float_info.max
        current_loss = lowest_loss

        while current_steps_without_progression < self.max_steps_without_progression:
            self._print_step_information(current_epochs,
                                         current_steps_without_progression,
                                         current_loss)

            model.train(self.training_data)
            current_epochs = current_epochs + 1
            current_loss, _ = model.test(self.validation_data)

            self.result.collect(model)

            if (current_loss < lowest_loss):
                current_steps_without_progression = 0
                self._remove_saved_models()
                save(model, SAVED_MODEL_NAME_FORMAT.format(current_epochs))
                optimal_epochs = current_epochs
                lowest_loss = current_loss
            else:
                current_steps_without_progression = current_steps_without_progression + 1

        model = load(SAVED_MODEL_NAME_FORMAT.format(optimal_epochs))
        print('    - optimal epochs {}'.format(optimal_epochs))
        self._remove_saved_models()
        return model

    def _remove_saved_models(self):
        os.system('rm ' + SAVED_MODEL_NAME_FORMAT.format('*'))

    def _print_step_information(self, current_epochs,
                                current_steps_without_progression,
                                current_loss):
        print(
            '\n--------------------- Neural Network Training Step ---------------------'
        )
        print('    - current epochs {}'.format(current_epochs))
        print('    - current steps without progression {}'.format(
            current_steps_without_progression))
        print('    - current loss {0:1.9f}'.format(current_loss))
