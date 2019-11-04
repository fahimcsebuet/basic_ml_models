import cvxopt
import cvxopt.solvers
import generate_random_data as grd
import numpy as np

class SVM(object):
    def __init__(self, C=1000.1):
        self.C = float(C)

    def kernel(self, x1, x2):
        return np.dot(x1, x2)

    def fit(self, X, y):
        sample_size, feature_size = X.shape

        # Generate Gram matrix K
        K = np.zeros((sample_size, sample_size))
        for i in range(0, sample_size):
            for j in range(0, sample_size):
                K[i, j] = self.kernel(X[i], X[j])

        # Mapping P, q, A, b, G and h to the dual parameters
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(sample_size) * (-1))
        A = cvxopt.matrix(y, (1, sample_size))
        b = cvxopt.matrix(0.0)
        # Calculate G and h for soft margin SVM
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(sample_size) * -1), np.identity(sample_size))))
        h = cvxopt.matrix(np.hstack((np.zeros(sample_size), np.ones(sample_size) * self.C)))

        # Solve QP optimization problem
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Extract lagrangian multiplier array
        a = np.ravel(sol['x'])

        # Find support vectors
        is_sv = a > 1e-5 # True if Lagrangian is non-zero
        ids = np.arange(len(a))[is_sv]
        self.sv_a = a[is_sv]
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]

        # Find the boundary intercept
        self.b = 0
        for i in range(len(self.sv_a)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.sv_a * self.sv_y * K[ids[i], is_sv])
        self.b /= len(self.sv_a)

        #Find weight vector
        self.w = np.zeros(feature_size)
        for i in range(len(self.sv_a)):
            self.w += self.sv_a[i] * self.sv_y[i] * self.sv_X[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X, threshold):
        return np.sign(self.project(X))
