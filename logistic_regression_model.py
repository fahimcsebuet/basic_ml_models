class LogisticRegression(object):

    def __init__(self, lam=0.01, iterations=100000):
        self.lam = lam
        self.iterations = iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            x = np.dot(X, self.theta)
            h = self.sigmoid(x)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lam * gradient

    def predict(self, X, threshold):
        if self.sigmoid(np.dot(X, self.theta)) >= threshold:
            return 1
        return -1
