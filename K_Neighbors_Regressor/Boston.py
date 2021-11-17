import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from LIB.plot import plot
from LIB.write_readme import add_readme

dir_ = "K_Neighbors_Regressor/"
answer = "# К ближайших соседей. Регрессия"
answer = add_readme("Задание", {"Текст": "В этом задании нужно проследить за изменением качества алгоритма К ближайших соседей в зависимости от изменений ***P*** праметра метрики Минковского. Данная метрика призванна для измерения расстояния между обьектами."},  answer)
answer = add_readme("Формула", {"Картинка": "Answer/img.png"}, answer)
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
answer = add_readme("Входные данные", {"Таблица": raw_df.head()}, answer)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

maxI, maxA = 0, -1000
history = [[], []]
for p in np.linspace(1, 10, 200):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric="minkowski")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(neigh, X, y, cv=cv, scoring='neg_mean_squared_error')
    history[0].append(p)
    history[1].append(scores.mean())
    if maxA < scores.mean():
        maxI = p
        maxA = scores.mean()
plot(history[0], history[1], title="Подбор p параметра\nв Метрике Минковского", label=("p", "Точность"), file=dir_+"Answer/h.png")
answer = add_readme("Измерения", {"Картинка": "Answer/h.png"}, answer)
answer = add_readme("Ответ", {"Текст": f"Большая точность при к = {maxI}: {maxA}"}, answer)
with open(dir_ + "README.md", "w") as readme:
    readme.write(answer)
