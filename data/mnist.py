import gzip
import os
import pickle

import numpy as np
import requests

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
OUT_DIR = '.mnist/'
OUT_FILE = 'mnist.pkl.gz'


def load():
    """Download the data set if not present localy."""
    print('Loading data ...')
    data_file = _open_data()
    training_data, validation_data, test_data = pickle.load(
        data_file, encoding='latin1')
    data_file.close()

    training_images = __format_images(training_data[0])
    training_labels = __format_labels(training_data[1])
    training_data = (training_images, training_labels)

    validation_images = __format_images(validation_data[0])
    validation_labels = __format_labels(validation_data[1])
    validation_data = (validation_images, validation_labels)

    test_images = __format_images(test_data[0])
    test_labels = __format_labels(test_data[1])
    test_data = (test_images, test_labels)

    return (training_data, validation_data, test_data)


def _open_data():
    try:
        return gzip.open(OUT_DIR + OUT_FILE, 'rb')
    except IOError:
        return _download_data()


def _download_data():
    print('Downloading data from {} ...'.format(DATA_URL))
    if not os.path.exists(os.path.join(os.curdir, OUT_DIR + OUT_FILE)):
        os.makedirs('.mnist', exist_ok=True)
        response = requests.get(DATA_URL)
        with open(OUT_DIR + OUT_FILE, "wb") as file:
            file.write(response.content)
    return gzip.open(OUT_DIR + OUT_FILE, 'rb')


def __format_labels(labels):
    formated_labels = []
    for label in labels:
        formated_labels.append(__format_label(label))
    return formated_labels


def __format_images(images):
    formated_images = []
    for image in images:
        formated_images.append(__format_image(image))
    return formated_images


def __format_image(image):
    return np.reshape(image, (784, 1))


def __format_label(label):
    formated_label = np.zeros((10, 1))
    formated_label[label] = 1.0
    return formated_label
