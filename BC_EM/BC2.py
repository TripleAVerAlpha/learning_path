# Теперь у нас есть данные не только о диаметре изделий, но и о, например, весе каждого изделия.
# Пускай станок №1 производит детали весом в 12 г. и стандартным отклонением в 1 г.,
# станок №2 производит изделия весом в 10 г. и стандартным отклонением 0.8 мм
import pandas as pd
import numpy as np
import math

ALL_DETAILS = 8000
N1 = 6000
N2 = 4000
MU = [[64, 14], [52, 9.5]]
SIGMA = np.array(([[[3.5, 0], [0, 1]], [[2, 0], [0, 0.7]]]))
FILE_PATH = "Data/"

k = 2
w = np.array([float(1. / k), float(1. / k)])


def gaus_func_02(k, m, x, mu, sigma):
    pj_xi = []
    for j in range(k):
        det_sigma_j = np.linalg.det(sigma[j])
        factor_1 = 1 / (((2 * math.pi) ** (k / 2)) * ((det_sigma_j) ** 0.5))
        factor_2 = []
        for i in x:
            factor_2.append(
                math.exp(-0.5 * np.matrix(i - mu[j]) * np.matrix(np.linalg.inv(sigma[j])) * np.matrix(i - mu[j]).T))
        pj_xi.append(factor_1 * np.array(factor_2))
    return np.array(pj_xi)


def proba_func_02(pjxi, w, k):
    P_X = []
    for j in range(k):
        P_X.append(w[j] * pjxi[j])
    P_X = np.sum(np.array(P_X), axis=0)
    P_J_X = []
    for j in range(k):
        P_J_X.append(w[j] * pjxi[j] / P_X)
    return np.array(P_J_X)


def pred_x_02(proba_X, limit_proba):
    pred_X = []
    for x in proba_X[0]:
        if x >= limit_proba:
            pred_X.append(1)
        else:
            pred_X.append(2)
    return np.array(pred_X)


detailsTable = pd.read_csv(FILE_PATH + "details.csv", index_col=0)
X = np.array([detailsTable['Диаметр'], detailsTable['Масса']]).T
pj_xi = gaus_func_02(k, 2, X, MU, SIGMA)
print(pj_xi)
proba_X = proba_func_02(pj_xi, w, k)
limit_proba = 0.5
pred_X = pred_x_02(proba_X, limit_proba)
detailsTable['Прогноз'] = pred_X
detailsTable['Результат'] = detailsTable['Прогноз'] == detailsTable['Выход']
count = detailsTable[detailsTable['Результат'] == True]['Результат'].count()
print("\rТочность: {:.2%}".format(count / ALL_DETAILS))
