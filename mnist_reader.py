from mlxtend.data import loadlocal_mnist
import numpy as np


def filter_0_or_1(in_X, in_y):
    X = []
    y = []
    for id in range(len(in_X)):
            if in_y[id] == 0 or in_y[id] == 1:
                X.append(in_X[id])
                y.append(in_y[id])
    return np.array(X), np.array(y)


def read_mnist_train_data():
    train_X, train_y = loadlocal_mnist(images_path='mnist/train-images-idx3-ubyte/data', labels_path='mnist/train-labels-idx1-ubyte/data')
    return filter_0_or_1(train_X, train_y)


def read_mnist_test_data():
    test_X, test_y = loadlocal_mnist(images_path='mnist/t10k-images-idx3-ubyte/data', labels_path='mnist/t10k-labels-idx1-ubyte/data')
    return filter_0_or_1(test_X, test_y)


def read_mnist_data():
    train_X, train_y = read_mnist_train_data()
    test_X, test_y = read_mnist_test_data()
    print(train_X.shape)
    print(train_y.shape)
    print(test_X.shape)
    print(test_y.shape)
    return train_X, train_y, test_X, test_y
