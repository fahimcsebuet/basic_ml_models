import numpy as np

def generate_class_a_points():
    # method to generate class A points corresponding to +1
    mean = [-1, -1]
    covariance_matrix = [[1, 1], [1, 1]]
    number_of_points = 200
    x, y = np.random.multivariate_normal(mean, covariance_matrix, number_of_points).T
    return zip(x, y)

def generate_class_b_points():
    # method to generate class A points corresponding to -1
    mean = [1, 1]
    covariance_matrix = [[1, 1], [1, 1]]
    number_of_points = 200
    x, y = np.random.multivariate_normal(mean, covariance_matrix, number_of_points).T
    return zip(x, y)
