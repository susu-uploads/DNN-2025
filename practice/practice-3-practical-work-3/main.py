import mnist_loader
import network2

if __name__ == '__main__':
    # загружаем данные MNIST
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # Создаем нейронную сеть
    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
    # Обучаем нейронную сеть
    net.SGD(training_data, 30, 10, 0.5,
            lmbda = 5.0,evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)
