import generate_random_data as grd
import linear_regression_model as lrm
import logistic_regression_model as lgrm
import matplotlib.pyplot as plt
import mnist_reader as mr
import numpy as np
from svm_model import SVM
import sys


def plot_svm(X_a, X_b, svm):
    # plot data points and suport vectors
    plt.style.use('classic')
    plt.title("Support Vector Machine C=" + str(svm.C))
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

    plt.savefig('svm_model'+ str(svm.C) +'.pdf', dpi=300, bbox_inches='tight')
    plt.clf()


def plot_regression(X_a, X_b, model, model_name):
    # plot data points and suport vectors
    plt.style.use('classic')
    plt.title(model_name)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.plot(X_a[:,0], X_a[:,1], 'rx', label='Class A')
    plt.plot(X_b[:,0], X_b[:,1], 'gx', label='Class B')

    # plot boundaries
    X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = model.project(X).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='black', linewidths=2, origin='lower')

    plt.legend(loc='best')

    plt.savefig(model_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.clf()


def main():
    X, y, X_a, X_b = grd.generate_dataset()

    C_list = [0.0001, 10.1, 100.1, 1000.1]
    for c in C_list:
        svm = SVM(c)
        svm.fit(X, y)
        plot_svm(X_a, X_b, svm)
        misclassification_error = np.sum(svm.predict(X) != y)/len(y)
        print(misclassification_error)

    mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y = mr.read_mnist_data()
    svm_mnist = SVM()
    svm_mnist.fit(mnist_train_X, mnist_train_y)
    generalization_error = np.sum(svn_mnist.predict(mnist_test_X) != mnist_test_y)/len(mnist_test_y)
    print(generalization_error)

    linear_reg = lrm.LinearRegression()
    linear_reg.fit(X, y)
    misclassification_error = np.sum(linear_reg.predict(X, 0) != y)/len(y)
    print(misclassification_error)
    plot_regression(X_a, X_b, linear_reg, "linear_regression")

    logistic_reg = lgrm.LogisticRegression()
    logistic_reg.fit(X, y)
    misclassification_error = np.sum(logistic_reg.predict(X, 0) != y)/len(y)
    print(misclassification_error)
    plot_regression(X_a, X_b, logistic_reg, "logistic_regression")


if __name__ == "__main__":
    main()
