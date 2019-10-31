import numpy as np
from pprint import pprint

def generate_class_a_points():
    # method to generate class A points corresponding to +1
    fsu_lib_id = 33290
    np.random.seed(fsu_lib_id)
    mean = [-1, -1]
    covariance_matrix = [[1, 0], [0, 1]]
    number_of_points = 200
    X = np.random.multivariate_normal(mean, covariance_matrix, number_of_points)
    labels = np.ones(number_of_points)
    return X, labels

def generate_class_b_points():
    # method to generate class B points corresponding to -1
    fsu_lib_id = 33290
    np.random.seed(fsu_lib_id)
    mean = [1, 1]
    covariance_matrix = [[1, 0], [0, 1]]
    number_of_points = 200
    X = np.random.multivariate_normal(mean, covariance_matrix, number_of_points)
    labels = np.ones(number_of_points) * (-1)
    return X, labels

def generate_dataset():
    X_a, labels_a = generate_class_a_points()
    X_b, labels_b = generate_class_b_points()
    return np.concatenate((X_a, X_b)), np.concatenate((labels_a, labels_b)), X_a, X_b
