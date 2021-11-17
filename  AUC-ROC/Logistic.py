import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

rez = []
for i in range(2):
    logistic = pd.read_csv('DataFrame/Logistic/data-logistic.csv', header=None)
    y = np.array([logistic[0]], dtype='float64')
    x = np.array(logistic[[1, 2]], dtype='float64').transpose()
    w = np.array([[100, -100]], dtype='float64')

    # print(x)
    # print(y)
    # print(w)

    k = 0.1
    C = 10.0
    error = 1
    N = 1.0 + np.exp(w.dot(x) * -y)
    wLast = np.array([[1, 1]], dtype='float64')
    dist = 10
    a = 0

    e = []
    while dist > 5:
        a += 1
        w[0][0] = np.sum(y*x[0]*(1.0 - 1.0/N))/len(y[0])*k + w[0][0] - i * k * C * w[0][0]
        w[0][1] = np.sum(y*x[1]*(1 - 1/N))/len(y[0])*k + w[0][1] - i * k * C * w[0][1]
        N = 1 + np.exp(w.dot(x) * -y)
        error = np.sum(np.log(N))/len(y[0]) + 1/2 * C * np.sum(np.power(w, 2))
        e.append(error)

        dist = (w - wLast) ** 2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)*100000
        wLast = w.copy()
        print(dist)
    rez.append(round(roc_auc_score(y[0], w.dot(x)[0]), 3))
print(w)
lib.dh(1, f"{rez[0]} {rez[1]}")
print(a)

