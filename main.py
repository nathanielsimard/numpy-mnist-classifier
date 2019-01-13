from classifier.nn import NeuralNetwork
from classifier.training import train
from data import mnist


def main():
    """Create and train a neural network with no hidden layer using early stopping regularization."""
    data = mnist.load()
    model = NeuralNetwork([784, 10], learning_rate=0.02, batch_size=100)

    result = train(model,
                   data.training_data,
                   data.test_data,
                   validation_data=data.validation_data,
                   early_stopping_regularization=True)
    result.save('models/mnist-1')


if __name__ == "__main__":
    main()
