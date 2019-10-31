import generate_random_data as grd
import matplotlib.pyplot as plt
import mnist_reader as mr
import numpy as np
from svm_model import SVM

def plot_svm(X_a, X_b, svm):
    # plot data points and suport vectors
    plt.style.use('classic')
    plt.title("Support Vector Machine")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.plot(X_a[:,0], X_a[:,1], 'rx', label='Class A')
    plt.plot(X_b[:,0], X_b[:,1], 'gx', label='Class B')
    plt.scatter(svm.sv_X[:,0], svm.sv_X[:,1], s=50, edgecolors='b', facecolors='none', label='Support Vector')

    # plot boundaries
    X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.project(X).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='black', linewidths=2, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='red', linewidths=1, linestyles = 'dashed', origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='green', linewidths=1, linestyles = 'dashed', origin='lower')

    plt.legend(loc='best')

    plt.savefig('svm_model.pdf', dpi=300, bbox_inches='tight')


def main():
    X, y, X_a, X_b = grd.generate_dataset()
    svm = SVM()
    svm.fit(X, y)
    plot_svm(X_a, X_b, svm)
    mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y = mr.read_mnist_data()
    svm_mnist = SVM()
    svm.fit(mnist_train_X, mnist_train_y)
    ret = svn.predict(mnist_test_X)
    print(ret)


if __name__ == "__main__":
    main()
