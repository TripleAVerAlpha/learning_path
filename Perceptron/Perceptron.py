import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from LIB.write_readme import add_readme

dir_ = "Perceptron/"
answer = "# Перцептрон. Нормализация признаков"
answer = add_readme("Задание", {"Текст": "В этом задании нужно проследить за изменением качества алгоритма К ближайших соседей в зависимости от изменений ***К*** и от нормализации данных"},  answer)
perceptron_test = pandas.read_csv(dir_+'Data/perceptron-test.csv', header=None)
perceptron_train = pandas.read_csv(dir_+'Data/perceptron-train.csv', header=None)
answer = add_readme("Входные данные", {"Таблица": perceptron_train.head()}, answer)


X_train, y_train = perceptron_train[[1, 2]], perceptron_train[0]
X_test, y_test = perceptron_test[[1, 2]], perceptron_test[0]

clf = Perceptron()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
noNorm = accuracy_score(y_test, predictions)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

perceptron_train[[1, 2]] = X_train_scaled
answer = add_readme("Нормализованные данные", {"Таблица": perceptron_train.head()}, answer)

clf = Perceptron()
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)
norm = accuracy_score(y_test, predictions)

answer = add_readme("Точность (accuracy) до нормализации", {"Текст": f"{noNorm:.2%}"},  answer)
answer = add_readme("Точность (accuracy) после нормализации", {"Текст": f"{norm:.2%}"},  answer)
answer = add_readme("Повышение точности", {"Текст": f"{(norm-noNorm):.3%}"},  answer)
with open(dir_ + "README.md", "w") as readme:
    readme.write(answer)