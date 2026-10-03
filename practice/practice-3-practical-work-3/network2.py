"""
network2.py

Модуль создания и обучения нейронной сети для распознавания рукописных цифр на основе метода стохастического 
градиентного спуска для прямой нейронной сети и стоимостной функции на основе перекрестной энтропии,
регуляризации и улучшенного способа инициализации весов нейронной сети. 

Группа: ЕТ-128
ФИО: Бабушкин Михаил Вадимович
"""

#### Библиотеки
# Стандартные библиотеки
import json                                                 # библиотека для кодирования/декодирования данных/объектов Python
import random                                               # библиотека функций для генерации случайных значений
# Сторонние библиотеки
import numpy as np                                          # библиотека функций для работы с матрицами
# Внутренние библиотеки
from cost import CrossEntropyCost                           # функция стоимости
from utils import sigmoid, sigmoid_prime, vectorized_result # утилитарные функции


class Network(object):
    def __init__(  # конструктор класса
            # указатель на объект класса
            self,
            # список размеров слоев нейронной сети
            sizes,
            # стоимостная функция (по умолчанию будет использоваться функция перекрестной энтропии)
            cost=CrossEntropyCost
    ):
        self.num_layers = len(sizes)  # задаем количество слоев нейронной сети
        self.sizes = sizes  # задаем список размеров слоев нейронной сети
        self.default_weight_initializer()  # метод инициализации начальных весов связей и смещений по умолчанию
        self.cost = cost  # задаем стоимостную функцию

    def default_weight_initializer(self):  # метод инициализации начальных весов связей и смещений
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]  # задаем случайные начальные смещения

        # задаем случайные начальные веса связей
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]  # задаем случайные начальные смещения

        # задаем случайные начальные веса
        self.weights = [np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def accuracy(# Оценка прогресса в обучении
         self # Указатель на объект класса
         , data # Набор данных (выборка)
         , convert=False # Признак необходимости изменять формат представления результата работы нейронной сети
         ):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def total_cost(
         # Значение функции потерь по всей выборке
         self, # Указатель на объект класса
         data, # Набор данных (выборка)
         lmbda, # Параметр сглаживания L2-регуляризации
         convert=False # Признак необходимости изменять формат представления результата работы нейронной сети
         ):
        cost = 0.0
        data = list(data)
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def backprop(  # Алгоритм обратного распространения
            self,  # Указатель на объект класса
            x,  # Вектор входных сигналов
            y  # Ожидаемый вектор выходных сигналов
    ):
        # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Определение переменных
        # Выходные сигналы слоя (первоначально соответствует выходным сигналам 1-го слоя или входным сигналам сети)
        activation = x
        # Список выходных сигналов по всем слоям (первоначально содержит только выходные сигналы 1-го слоя)
        activations = [x]
        # Список активационных потенциалов по всем слоям (первоначально пуст)
        zs = []
        # Прямое распространение
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # Считаем активационные потенциалы текущего слоя
            zs.append(z)  # Добавляем элемент (активационные потенциалы слоя) в конец списка
            # Считаем выходные сигналы текущего слоя, применяя сигмоидальную функцию активации к активационным потенциалам слоя
            activation = sigmoid(z)
            activations.append(activation)  # Добавляем элемент (выходные сигналы слоя) в конец списка
        # Обратное распространение
        # Считаем меру влияния нейронов выходного слоя L на величину ошибки (BP1)
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta  # Градиент dC/db для слоя L (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  # Градиент dC/dw для слоя L (BP4)
        for l in range(2, self.num_layers):
            z = zs[-l]  # Активационные потенциалы l-го слоя (двигаемся по списку справа налево)
            sp = sigmoid_prime(z)  # Считаем сигмоидальную функцию от активационных потенциалов l-го слоя
            # Считаем меру влияния нейронов l-го слоя  на величину ошибки (BP2)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta  # Градиент dC/db для l-го слоя (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)  # Градиент dC/dw для l-го слоя (BP4)

    def update_mini_batch(  # Шаг градиентного спуска
            self,  # указатель на объект класса
            mini_batch,  # подвыборка
            eta,  # скорость обучения
            lmbda,  # параметр сглаживания L2-регуляризации
            n  #
    ):
        # список градиентов dC/db для каждого слоя (первоначально заполняются нулями)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # список градиентов dC/dw для каждого слоя (первоначально заполняются нулями)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # послойно вычисляем градиенты dC/db и dC/dw для текущего прецедента (x, y)
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # суммируем градиенты dC/db для различных прецедентов текущей подвыборки
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            # суммируем градиенты dC/dw для различных прецедентов текущей подвыборки
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # обновляем все веса w нейронной сети
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        # обновляем все смещения b нейронной сети
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            # параметр сглаживания L2-регуляризации
            lmbda=0.0,
            # оценочная выборка
            evaluation_data=None,
            # флаг вывода на экран информа-ции о значении стоимостной функции в процессе обучения,
            # рассчитанном на оценочной выборке
            monitor_evaluation_cost=False,
            # флаг вывода на экран ин-формации о достигнутом прогрессе в обучении, рассчитанном на оценочной выборке
            monitor_evaluation_accuracy=False,
            # флаг вывода на экран информации о значении стоимостной функции в процессе обучения,
            # рассчитанном на обучающей выборке
            monitor_training_cost=False,
            # флаг вывода на экран инфор-мации о достигнутом прогрессе в обучении, рассчитанном на обучающей выборке
            monitor_training_accuracy=False
    ):
        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        training_data = list(training_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size]
                            for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("--Cost on training data: {}".format(cost))

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("--Accuracy on training data: {} / {}".format(accuracy, n))

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("--Cost on evaluation data: {}".format(cost))

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("--Accuracy on evaluation data: {} / {}".format(
                    self.accuracy(evaluation_data), n_data))

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def save(self, filename): # Запись нейронной сети в файл
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
