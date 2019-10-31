from mlxtend.data import loadlocal_mnist

train_X, train_y = loadlocal_mnist(images_path='mnist/train-images-idx3-ubyte/data', labels_path='mnist/train-labels-idx1-ubyte/data')
test_X, test_y = loadlocal_mnist(images_path='mnist/t10k-images-idx3-ubyte/data', labels_path='mnist/t10k-labels-idx1-ubyte/data')

print(test_X.shape)
print(test_y.shape)
