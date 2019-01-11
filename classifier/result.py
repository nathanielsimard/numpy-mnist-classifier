import matplotlib.pyplot as plt
from nn import NeuralNetwork

from typing import List


class Result():
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data
        self.training_accuracies = []
        self.test_accuracies = []
        self.training_losses = []
        self.test_losses = []

    def collect(self, model: NeuralNetwork):
        test_loss, test_accuracy = model.test(self.test_data)
        training_loss, training_accuracy = model.test(self.training_data)

        print(
            '\n------------------ Neural Network Testing Result ------------------\n'
            + '   - epoch {0} \n'.format(len(self.test_accuracies)) +
            '   - test loss {0:1.9f}\n'.format(test_loss) +
            '   - test accuracy {0:1.3f} %\n'.format(test_accuracy) +
            '   - training loss {0:1.9f}\n'.format(training_loss) +
            '   - training accuracy {0:1.3f} %\n'.format(training_accuracy))

        self.test_accuracies.append(test_accuracy / 100.0)
        self.training_accuracies.append(training_accuracy / 100.0)
        self.test_losses.append(test_loss)
        self.training_losses.append(training_loss)

    def plot(self, file_name: str):
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
