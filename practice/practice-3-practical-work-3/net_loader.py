"""
net_loader.py

Модуль, содержащий функционал для загрузки нейронной сети из файла.

Группа: ЕТ-128
ФИО: Бабушкин Михаил Вадимович
"""
# Библиотеки
# Стандартные библиотеки
import json                  # библиотека для кодирования/декодирования данных/объектов Python
import sys                   # библиотека для работы с переменными и функциями, имеющими отношение к интерпретатору и его окружению
# Сторонние библиотеки
import numpy as np           # библиотека для работы с матрицами
# Внутренние библиотеки
import cost as cst           # модуль стоимостных функций
from network2 import Network # класс нашей нейронной сети v2


def load(filename):  # Загрузка нейронной сети из файла
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[cst.__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
