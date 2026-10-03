"""
cost.py

Модуль, содержащий описания стоимостных функций.

Группа: ЕТ-128
ФИО: Бабушкин Михаил Вадимович
"""
# Библиотеки
# Сторонние библиотеки
import numpy as np              # библиотека для работы с матрицами
# Внутренние библиотеки
from utils import sigmoid_prime # производная сигмоидальной функции


class QuadraticCost(object):  # Определение среднеквадратичной стоимостной функции

    @staticmethod
    def fn(a, y):  # Cтоимостная функция
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):  # Мера влияния нейронов выходного слоя на величину ошибки
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):  # Определение стоимостной функции на основе перекрестной энтропии

    @staticmethod
    def fn(a, y):  # Cтоимостная функция
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):  # Мера влияния нейронов выходного слоя на величину ошибки
        return (a - y)
