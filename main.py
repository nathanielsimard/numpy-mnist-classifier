from classifier import nn, training
from data import mnist


def main():
    """Create a Neural Network with one hidden layer."""
    training_data, validation_data, test_data = mnist.load()

    model = nn.NeuralNetwork([784, 100, 10], learning_rate=0.01, batch_size=50)

    model_training = training.EarlyStoppingRegularization(model,
                                                          training_data,
                                                          validation_data,
                                                          test_data,
                                                          max_steps_without_progression=2)
    result = model_training.train()

    result.save('models/mnist')


if __name__ == "__main__":
    main()
