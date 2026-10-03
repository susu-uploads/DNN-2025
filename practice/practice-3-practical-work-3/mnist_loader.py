"""
mnist_loader.py

Модуль для подключения и использования базы данных MNIST.
Группа: ЕТ-128
ФИО: Бабушкин Михаил Вадимович
"""
# Библиотеки
# Стандартные библиотеки
import gzip                         # библиотека для сжатия и распаковки файлов gzip и gunzip
import pickle                       # библиотека для сохранения и загрузки сложных объектов Python
# Сторонние библиотеки
import numpy as np                  # библиотека для работы с матрицами
# Внутренние библиотеки
from utils import vectorized_result # метод преобразования числовой метки в выходной массив


def load_data(): # Загружает и возвращает наборы данных MNIST для обучения, валидации и тестирования.
    f = gzip.open('mnist.pkl.gz', 'rb')  # открываем сжатый файл gzip в двоичном режиме
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')  # загружаем таблицы из файла
    f.close()  # закрываем файл
    return training_data, validation_data, test_data

def load_data_wrapper(): # Подготавливает данные MNIST для подачи в нейронную сеть в формате пар (вход, выход).
    # инициализация наборов данных в формате MNIST
    tr_d, va_d, te_d = load_data()
    # преобразование массивов размера 1×784 к массивам размера 784×1
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # представление цифр от 0 до 9 в виде массивов размера 10×1
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # формируем набор обучающих данных из пар (x, y)
    training_data = zip(training_inputs, training_results)
    # преобразование массивов размера 1×784 к массивам размера 784×1
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    # формируем набор данных проверки из пар (x, y)
    validation_data = zip(validation_inputs, va_d[1])
    # преобразование массивов размера 1×784 к массивам размера 784×1
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    # формируем набор тестовых данных из пар (x, y)
    test_data = zip(test_inputs, te_d[1])
    # возвращаем набор данных
    return training_data, validation_data, test_data
