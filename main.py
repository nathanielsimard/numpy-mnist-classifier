from classifier.nn import NeuralNetwork
from classifier.training import train
from data import mnist


def main():
    """Create and train a neural network using early stopping regularization."""
    data = mnist.load()
    model = NeuralNetwork([784, 150, 10], learning_rate=0.01, batch_size=100)

    result = train(model,
                   data.training_data,
                   data.test_data,
                   validation_data=data.validation_data,
                   epochs=2,
                   early_stopping_regularization=True)
    result.save('models/mnist-2')


if __name__ == "__main__":
    main()
