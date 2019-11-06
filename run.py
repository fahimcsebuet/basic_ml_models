import generate_random_data as grd
import linear_regression_model as lrm
import logistic_regression_model as lgrm
import matplotlib.pyplot as plt
import mnist_reader as mr
import numpy as np
from svm_model import SVM
import sys


def plot_svm(X_a, X_b, svm, model_name):
    # plot data points and suport vectors
    plt.style.use('classic')
    plt.title(model_name + " C=" + str(svm.get_C()))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.plot(X_a[:,0], X_a[:,1], 'rx', label='Class A')
    plt.plot(X_b[:,0], X_b[:,1], 'gx', label='Class B')
    plt.scatter(svm.sv_X[:,0], svm.sv_X[:,1], s=50, edgecolors='b', facecolors='none', label='Support Vector')

    # plot boundaries
    boundary_range = [-6, 6]
    if svm.get_C() == 0.0001:
        boundary_range = [-25, 25]
    X1, X2 = np.meshgrid(np.linspace(boundary_range[0], boundary_range[1], 50), np.linspace(boundary_range[0], boundary_range[1], 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.project(X).reshape(X1.shape)

    plt.contour(X1, X2, Z, [0.0], colors='black', linewidths=2, origin='lower')
    plt.contour(X1, X2, Z + 1, [0.0], colors='green', linewidths=1, linestyles = 'dashed', origin='lower')
    plt.contour(X1, X2, Z - 1, [0.0], colors='red', linewidths=1, linestyles = 'dashed', origin='lower')

    plt.legend(loc='best')

    plt.savefig(model_name + str(svm.C) +'.pdf', dpi=300, bbox_inches='tight')
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

    threshold = 0.0
    if model.name() == "logistic":
        threshold = 0.5

    plt.contour(X1, X2, Z-threshold, [0.0], colors='black', linewidths=2, origin='lower')

    plt.legend(loc='best')

    plt.savefig(model_name + '.pdf', dpi=300, bbox_inches='tight')
    plt.clf()


def run_mnist(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, model):
    model.fit(mnist_train_X, mnist_train_y)
    margin = 0.0
    try:
        margin = model.margin()
    except:
        margin = 0.0

    threshold = 0.0
    if model.name() == "logistic":
        threshold = 0.5

    generalization_error = np.sum(model.predict(mnist_test_X, threshold) != mnist_test_y)/len(mnist_test_y)
    return generalization_error, margin


def run_model(X, y, X_a, X_b, model, model_name, plot_func):
    model.fit(X, y)
    margin = 0.0
    try:
        margin = model.margin()
    except:
        margin = 0.0

    plot_func(X_a, X_b, model, model_name)

    threshold = 0.0
    if model.name() == "logistic":
        threshold = 0.5

    misclassification_error = np.sum(model.predict(X, threshold) != y)/len(y)
    return misclassification_error, margin


def run_dummy_data_training(out_file):
    X, y, X_a, X_b = grd.generate_dataset()

    C_list = [0.0001, 0.001, 100.1, 1000.1]
    for c in C_list:
        svm = SVM(c)
        misclassification_error, margin = run_model(X, y, X_a, X_b, svm, "support_vector_machine", plot_svm)
        out_file.write("SVM C:" + str(c) + " Misclassification Error=" + str(misclassification_error) + " Margin=" + str(margin) + "\n")

    linear_reg = lrm.LinearRegression()
    misclassification_error, margin = run_model(X, y, X_a, X_b, linear_reg, "linear_regression", plot_regression)
    out_file.write("Linear Regression Misclassification Error=" + str(misclassification_error) + " Margin=" + str(margin) + "\n")

    logistic_reg = lgrm.LogisticRegression()
    misclassification_error, margin = run_model(X, y, X_a, X_b, logistic_reg, "logistic_regression", plot_regression)
    out_file.write("Logistic Regression Misclassification Error=" + str(misclassification_error) + " Margin=" + str(margin) + "\n")


def run_mnist_training(out_file):
    mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y = mr.read_mnist_data()

    svm_mnist = SVM()
    generalization_error, margin = run_mnist(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, svm_mnist)
    out_file.write("SVM MNIST C:" + str(svm_mnist.get_C()) + " Generalization Error=" + str(generalization_error) + " Margin=" + str(margin) + "\n")

    linear_reg = lrm.LinearRegression()
    generalization_error, margin = run_mnist(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, linear_reg)
    out_file.write("Linear Regression Generalization Error=" + str(generalization_error) + " Margin=" + str(margin) + "\n")

    logistic_reg = lgrm.LogisticRegression()
    generalization_error, margin = run_mnist(mnist_train_X, mnist_train_y, mnist_test_X, mnist_test_y, logistic_reg)
    out_file.write("Logistic Regression Generalization Error=" + str(generalization_error) + " Margin=" + str(margin) + "\n")


def run_model_with_cvloo(X, y, model):
    train_X = X
    train_y = y
    test_X = []
    test_y = []
    cv_error = 0.0
    for id in range(len(X)):
        test_X.append(train_X[id])
        test_X = np.array(test_X).astype(np.double)
        test_y.append(train_y[id])
        test_y = np.array(test_y).astype(np.double)
        train_X = np.delete(train_X, id, 0)
        train_y = np.delete(train_y, id, 0)
        model.fit(train_X, train_y)
        threshold = 0.0
        if model.name() == "logistic":
            threshold = 0.5
        cv_error += np.sum(model.predict(test_X, threshold) != test_y)/len(test_y)
        train_X = X
        train_y = y
        test_X = []
        test_y = []
    cv_error /= len(y)
    return cv_error


def run_dummy_data_cvloo_training(out_file):
    X, y, X_a, X_b = grd.generate_dataset()

    C_list = [0.0001, 0.001, 100.1, 1000.1]
    for c in C_list:
        svm = SVM(c)
        cv_error = run_model_with_cvloo(X, y, svm)
        out_file.write("SVM C:" + str(c) + " Cross-validation Error=" + str(cv_error) + "\n")

    linear_reg = lrm.LinearRegression()
    cv_error = run_model_with_cvloo(X, y, linear_reg)
    out_file.write("Linear Regression Cross-validation Error=" + str(cv_error) + "\n")

    logistic_reg = lgrm.LogisticRegression()
    cv_error = run_model_with_cvloo(X, y, logistic_reg)
    out_file.write("Logistic Regression Cross-validation Error=" + str(cv_error) + "\n")


def main():
    out_file = open("run.out", "w")
    run_dummy_data_training(out_file)
    run_mnist_training(out_file)
    run_dummy_data_cvloo_training(out_file)
    out_file.close()


if __name__ == "__main__":
    main()
