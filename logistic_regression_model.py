import numpy as np

class LogisticRegression(object):

    def __init__(self, lam=0.001, iterations=10000):
        self.lam = lam
        self.iterations = iterations

    def name(self):
        return "logistic"

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, h, y):
        return (-y * np.log(h+1e-10) -(1 - y) * np.log(1 - h+1e-10)).mean()

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        prev_loss = 1000000.0
        loss_threshold = 1e-5
        for _ in range(self.iterations):
            x = np.dot(X, self.theta)
            h = self.sigmoid(x)
            curr_loss = self.loss(h, y)
            if np.abs(curr_loss - prev_loss) < loss_threshold:
                break
            prev_loss = curr_loss
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lam * gradient

    def project(self, X):
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold):
        ret = []
        for proj in self.project(X):
            if proj > threshold:
                ret.append(1)
            else:
                ret.append(-1)
        return np.array(ret)
