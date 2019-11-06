import numpy as np


class LinearRegression:

    def __init__(self, lam=0.001, iterations=10000):
        self.lam = lam
        self.iterations = iterations

    def name(self):
        return "linear"

    def loss(self, y_pred, y):
        return np.sum(np.square(np.subtract(y_pred%(2**300), y%(2**300)))) / (2 * y.shape[0])

    def fit(self, X, y):
        self.w = np.zeros((X.shape[1], 1))
        y = y.reshape(len(y), 1)
        m = X.shape[0]

        prev_loss = 1000000.0
        loss_threshold = 1e-5
        for _ in range(self.iterations):
            y_pred = np.dot(X, self.w)
            curr_loss = self.loss(y_pred, y)
            if np.abs(curr_loss - prev_loss) < loss_threshold:
                break
            prev_loss = curr_loss
            residuals = y_pred - y
            gradients = np.dot(X.T, residuals)
            self.w -= ((self.lam / m) * gradients)

    def project(self, X):
        return np.dot(X, self.w)

    def predict(self, X, threshold):
        ret = []
        for proj in self.project(X):
            if proj[0] >= threshold:
                ret.append(1)
            else:
                ret.append(-1)
        return np.array(ret)
