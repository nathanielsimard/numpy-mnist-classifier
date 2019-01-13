import os

import matplotlib.pyplot as plt
import numpy as np
from .nn import NeuralNetwork, save


class ResultCollector():
    """Collect data from model by running test on different data set."""

    def __init__(self):
        self.training_accuracies = []
        self.test_accuracies = []
        self.training_losses = []
        self.test_losses = []
        self.epochs = 0

    def collect(self,
                epoch,
                test_loss,
                test_accuracy,
                training_loss,
                training_accuracy):
        """Collect loss and accuracy on trainind and test data set.

        :param model: The model to be tested
        """
        print(
            '\n------------------ Neural Network Testing Result ------------------\n',
            '   - epoch {0} \n'.format(epoch),
            '   - test loss {0:1.9f}\n'.format(test_loss),
            '   - test accuracy {0:1.3f} %\n'.format(test_accuracy),
            '   - training loss {0:1.9f}\n'.format(training_loss),
            '   - training accuracy {0:1.3f} %\n'.format(training_accuracy))

        self.test_accuracies.append(test_accuracy / 100.0)
        self.training_accuracies.append(training_accuracy / 100.0)
        self.test_losses.append(test_loss)
        self.training_losses.append(training_loss)
        if self.epochs < epoch:
            self.epochs = epoch


class Result():
    """Produce a report containing training result."""

    FORMAT = '# Result'\
        + '\n'\
        + '\nTrained the model for {0} epochs.'\
        + '\n'\
        + '\n## Model'\
        + '\n'\
        + '\n- Layers : {5}'\
        + '\n- Activation : {6}'\
        + '\n- Learning Rate : {9}'\
        + '\n- Batch Size : {10}'\
        + '\n'\
        + '\n## Data'\
        + '\n'\
        + '\nSize :'\
        + '\n'\
        + '\n- Training : {7}'\
        + '\n- Test : {8}'\
        + '\n- Validation : {11}'\
        + '\n'\
        + '\n### Sample'\
        + '\n'\
        + '\n![graph](./sample.png)'\
        + '\n'\
        + '\n## Accuracy and Loss'\
        + '\n'\
        + '\n|   | Training | Test |'\
        + '\n|---|---|---|'\
        + '\n| Accuracy | {1:1.3f}% | {2:1.3f}%  |'\
        + '\n| Loss | {3:1.3f} | {4:1.3f} |'\
        + '\n'\
        + '\n![graph](./result.png)'\


    def __init__(self,
                 training_data,
                 test_data,
                 validation_data,
                 training_accuracies,
                 test_accuracies,
                 training_losses,
                 test_losses,
                 epochs,
                 model: NeuralNetwork):
        self.training_data = training_data
        self.test_data = test_data
        self.validation_data = validation_data

        self.training_accuracies = training_accuracies
        self.training_losses = training_losses

        self.test_accuracies = test_accuracies
        self.test_losses = test_losses

        self.epochs = epochs
        self.model = model

    def save(self, directory_name: str):
        """Produce a report containing training result."""
        os.system('mkdir -p {}'.format(directory_name))
        self._plot_accuracy()
        self._plot_losses()
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(directory_name + '/result.png')
        plt.close()

        self._plot_sample(self.test_data)
        plt.savefig(directory_name + '/sample.png')
        plt.close()

        os.system('touch {}/result.md'.format(directory_name))

        content = self.FORMAT.format(self.epochs,
                                     self.training_accuracies[-1] * 100,
                                     self.test_accuracies[-1] * 100,
                                     self.training_losses[-1],
                                     self.test_losses[-1],
                                     self.model.layers_size,
                                     self.model.activation,
                                     len(self.training_data[0]),
                                     len(self.test_data[0]),
                                     self.model.learning_rate,
                                     self.model.batch_size,
                                     len(self.validation_data[0]))
        os.system('echo "{0}" > {1}/result.md'.format(content, directory_name))
        save(self.model, '{}/model.pkl'.format(directory_name))

    def _plot_sample(self, sample_data):
        columns = 8
        rows = 8
        fig = plt.figure(figsize=(6, 6))

        for i in range(1, rows * columns + 1):
            img = np.reshape(sample_data[0][i], (28, 28))
            plot = fig.add_subplot(rows, columns, i)
            plot.imshow(img)
            plot.axis('off')

    def _plot_losses(self):
        plt.plot(self.training_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Test Loss')

    def _plot_accuracy(self):
        plt.plot(self.training_accuracies, label='Training Accuracy')
        plt.plot(self.test_accuracies, label='Test Accuracy')
