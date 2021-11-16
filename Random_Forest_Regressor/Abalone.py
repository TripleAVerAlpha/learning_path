import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from LIB import plot

# -------------------------------------------Настройка директорий-------------------------------------------------------
dir_ = "Random_Forest_Regressor/"


# --------------------------------------Загрузка и поготовка датасета---------------------------------------------------
abalone = pandas.read_csv(dir_ + 'Data/abalone.csv')
abalone['Sex'] = abalone['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = np.array(abalone.loc[:, :abalone.columns[len(abalone.columns)-2]])
y = np.array(abalone.loc[:, abalone.columns[len(abalone.columns)-1]])


# ------------------------------------------Подбор обучение модели------------------------------------------------------
all_score = []
kol_tree = 50
for i in range(1, kol_tree+1):
    print(f"Деревьев = {i}")
    clf = RandomForestRegressor(n_estimators=i, random_state=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    score = cross_val_score(clf, X, y, cv=cv, scoring='r2')
    all_score.append(round(score.mean(), 2))


# -------------------------------------------Обработка результата-------------------------------------------------------
try:
    min_N = min((np.array(all_score) > 0.5).nonzero()[0]) + 1
    print(f"Минимальное кол-во деревьев, для точности больше 0,52: {min_N}")
except ValueError:
    min_N = "Не найдено"
    print(f"При {kol_tree} деревьях не существует точности больше 0,52")


# -----------------------------------------------Строим гравик----------------------------------------------------------
plot.plot(range(len(all_score)), all_score,
          title=f"Оптимальое кол-во: {min_N}",
          label=("Кол-во деревьев", "Точность"),
          file=dir_ + "Answer/answer.png")
