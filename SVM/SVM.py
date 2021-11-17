from sklearn import svm
import pandas as pd

from LIB.plot import plotSVM
from LIB.write_readme import add_readme

dir_ = "SVM/"
answer = "# Метод опорных векторов. Опорные объекты"
answer = add_readme("Задание", {"Текст": "Определите объекты которые разделяют выборку лучше всего."},  answer)
data = pd.read_csv(dir_ + 'Data/svm-data.csv', header=None)
data.columns = ["Цель", "x", "y"]
answer = add_readme("Входные данные", {"Таблица": data}, answer)


clf = svm.SVC(C=100000, kernel='linear')
clf.fit(data[["x", "y"]].values, data["Цель"].values)
a = " ".join(map(lambda x: str(x), clf.support_))
plotSVM(data.values, clf.support_, file=dir_+"Answer/plot.png")
answer = add_readme("Разделение", {"Картинка": "Answer/plot.png"}, answer)

answer = add_readme("Опорные объекты", {"Текст": f"{a}"},  answer)
with open(dir_ + "README.md", "w") as readme:
    readme.write(answer)