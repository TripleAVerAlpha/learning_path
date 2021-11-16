from sklearn.tree import DecisionTreeClassifier
import pandas
from LIB.write_readme import add_readme

dir_ = "Decision_Tree_Classifier/"
answer = "# Построение Решающего дерева и трактовка весов"
answer = add_readme("Задание", {"Текст": "Постройте Решающее дерево, определяющее выживет пассажир или нет. Что больше всего влияет на решение дерева?"}, answer)

titanic = pandas.read_csv('Pandas/Data/titanic.csv', index_col='PassengerId')
answer = add_readme("Входные данные", {"Таблица": titanic[['Survived', 'Pclass', 'Sex', 'Fare', 'Age']].head()}, answer)

titanic = titanic[['Survived', 'Pclass', 'Sex', 'Fare', 'Age']].dropna()
titanic.loc[(titanic.Sex == 'female'), 'Sex'] = 0
titanic.loc[(titanic.Sex == 'male'), 'Sex'] = 1
answer = add_readme("Обработанные данные", {"Таблица": titanic.head()}, answer)

clf = DecisionTreeClassifier(random_state=1)
input_ = ['Pclass', 'Sex', 'Fare', 'Age']
input_ru = ['Класс билета', 'Пол', 'Пассажирский тариф', 'Возраст']
clf.fit(titanic[input_], titanic['Survived'])
importances = clf.feature_importances_

max_1 = 0
max_2 = 0
for i in range(1, len(importances)):
    if importances[max_1] < importances[i]:
        max_1, max_2 = i, max_1

answer = add_readme("Ответ", {"Текст": "При построении классификационного дерева решения на данном датасете важны параметры:<br>"
                                       f"{input_ru[max_1]} ({input_[max_1]}), вес = {importances[max_1]}<br>"
                                       f"{input_ru[max_2]} ({input_[max_2]}), вес = {importances[max_2]}"}, answer)

with open(dir_ + "README.md", "w") as readme:
    readme.write(answer)
