import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from LIB.plot import plot, multiplot
from LIB.write_readme import add_readme

rez = []
dir_ = "AUC-ROC/"
answer = "# Метод опорных векторов. Опорные объекты"
answer = add_readme("Задание", {"Текст": "Определите объекты которые разделяют выборку лучше всего."}, answer)
logistic = pd.read_csv(dir_ + 'Data/data-logistic.csv', header=None)
answer = add_readme("Входные данные", {"Таблица": logistic}, answer)


def train(C, i, k):
    y = np.array([logistic[0]], dtype='float64')
    x = np.array(logistic[[1, 2]], dtype='float64').transpose()
    w = np.array([[0, 0]], dtype='float64')

    N = 1.0 + np.exp(w.dot(x) * -y)
    wLast = np.array([[1, 1]], dtype='float64')
    dist = 10
    d = []
    while dist > 5:
        w[0][0] = np.sum(y * x[0] * (1.0 - 1.0 / N)) / len(y[0]) * k + w[0][0] - i * k * C * w[0][0]
        w[0][1] = np.sum(y * x[1] * (1 - 1 / N)) / len(y[0]) * k + w[0][1] - i * k * C * w[0][1]
        N = 1 + np.exp(w.dot(x) * -y)
        dist = (w - wLast) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist) * 100000
        wLast = w.copy()
        d.append(dist[0])
    return d, round(roc_auc_score(y[0], w.dot(x)[0]), 3)


dist_array = []
for i in range(2):
    d, rez = train(10, i, 0.1)
    dist_array.append(d[1:])
dist_array = np.array(dist_array)
print(dist_array.shape)
multiplot(dist_array, file=dir_ + f"Answer/dist0.png")
print(f"{rez}")
