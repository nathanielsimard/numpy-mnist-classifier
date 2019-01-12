"""Training neural network models for MNIST classification."""

from classifier.nn import NeuralNetwork
from classifier.training import EarlyStoppingTraining
from data import mnist


def main():
    """Create and train a neural network with no hidden layer using early stopping regularization."""
    data = mnist.load()
    model = NeuralNetwork([784, 10], learning_rate=0.02, batch_size=100)

    training_early_stopping = EarlyStoppingTraining(model,
                                                    data.training_data,
                                                    data.test_data,
                                                    data.validation_data)
    training_early_stopping.train()
    training_early_stopping.save_result("layers-784-10_early-stopping-regularization")


if __name__ == "__main__":
    main()
