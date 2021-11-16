import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas
from LIB.plot import plot
from LIB.write_readme import add_readme

dir_ = "K_Neighbors_Classifier/"
answer = "# К ближайших соседей. Подбор параметров."
answer = add_readme("Задание", {"Текст": "В этом задании нужно проследить за изменением качества алгоритма К ближайших соседей в зависимости от изменений ***К*** и от нормализации данных"},  answer)

wine = pandas.read_csv(dir_+'Data/wine.data', header=None).sample(frac=1).reset_index(drop=True)
answer += "\n### До нормализации"
answer = add_readme("Входные данные", {"Таблица": wine.head()}, answer)

maxI, maxA = 0, 0
history = [[], []]
for i in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=i)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(neigh, np.array(wine[list(range(1, 14))]), wine[0], cv=cv, scoring='accuracy')
    history[0].append(i)
    history[1].append(scores.mean())
    if round(maxA, 4) < round(scores.mean(), 4):
        maxI = i
        maxA = scores.mean()
plot(history[0], history[1], "Результаты", label=("К - соседей", "Точность"), file=dir_+"Answer/h1.png")
answer = add_readme("График", {"Картинка": "Answer/h1.png"}, answer)
answer = add_readme("Ответ", {"Текст": f"Большая точность при к = {maxI}: {maxA}"}, answer)
scaler = preprocessing.StandardScaler().fit(wine[list(range(1, 14))])
wine[list(range(1, 14))] = scaler.transform(wine[list(range(1, 14))])
answer += "\n### После нормализации"
answer = add_readme("Нормализованные данные", {"Таблица": wine.head()}, answer)
maxI, maxA = 0, 0
history = [[], []]
for i in range(1, 50):
    neigh = KNeighborsClassifier(n_neighbors=i)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(neigh, wine[list(range(1, 14))], wine[0], cv=cv)
    history[0].append(i)
    history[1].append(scores.mean())
    if maxA < scores.mean():
        maxI = i
        maxA = scores.mean()
plot(history[0], history[1], "Результаты", label=("К - соседей", "Точность"), file=dir_+"Answer/h2.png")
answer = add_readme("График", {"Картинка": "Answer/h2.png"}, answer)
answer = add_readme("Ответ", {"Текст": f"Большая точность при к = {maxI}: {maxA}"}, answer)

answer = add_readme("Результат", {"Текст": f"Большая точность достигается при нормализации данных и при к = {maxI}: {maxA:.2%}"}, answer)

with open(dir_ + "README.md", "w") as readme:
    readme.write(answer)