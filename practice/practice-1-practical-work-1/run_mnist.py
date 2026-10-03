# run_mnist.py (скрипт для запуска, сохраните в той же директории, где network.py и mnist_loader.py, а также mnist.pkl.gz)

import mnist_loader
import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
