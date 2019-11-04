import numpy as np


class LinearRegression:

    def __init__(self, lam=0.05, iterations=1000):
        self.lam = lam
        self.iterations = iterations

    def fit(self, X, y):
        self.w = np.zeros((X.shape[1], 1))
        y = y.reshape(len(y), 1)
        m = X.shape[0]

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.w)
            residuals = y_pred - y
            gradients = np.dot(X.T, residuals)
            self.w -= ((self.lam / m) * gradients)

    def project(self, X):
        return np.dot(X, self.w)

    def predict(self, X, threshold):
        proj = self.project(X)
        if self.project(X).all() >= threshold:
            return 1
        return -1
