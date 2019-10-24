import numpy as np
from pprint import pprint

def generate_class_a_points():
    # method to generate class A points corresponding to +1
    fsu_lib_id = 33290
    np.random.seed(fsu_lib_id)
    mean = [-1, -1]
    covariance_matrix = [[1, 0], [0, 1]]
    number_of_points = 200
    x, y = np.random.multivariate_normal(mean, covariance_matrix, number_of_points).T
    return zip(x, y)

def generate_class_b_points():
    # method to generate class B points corresponding to -1
    fsu_lib_id = 33290
    np.random.seed(fsu_lib_id)
    mean = [1, 1]
    covariance_matrix = [[1, 0], [0, 1]]
    number_of_points = 200
    x, y = np.random.multivariate_normal(mean, covariance_matrix, number_of_points).T
    return zip(x, y)

