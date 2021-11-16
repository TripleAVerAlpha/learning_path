# Построение Решающего дерева и трактовка весов
**Задание:**<br>
Постройте Решающее дерево, определяющее выживет пассажир или нет. Что больше всего влияет на решение дерева?<br><br>
**Входные данные:**
<br><table><tr><th>Индексы</th><th>Survived</th><th>Pclass</th><th>Sex</th><th>Fare</th><th>Age</th></tr><tr><th>1</th><th>0</th><th>3</th><th>male</th><th>7.25</th><th>22.0</th></tr><tr><th>2</th><th>1</th><th>1</th><th>female</th><th>71.2833</th><th>38.0</th></tr><tr><th>3</th><th>1</th><th>3</th><th>female</th><th>7.925</th><th>26.0</th></tr><tr><th>4</th><th>1</th><th>1</th><th>female</th><th>53.1</th><th>35.0</th></tr><tr><th>5</th><th>0</th><th>3</th><th>male</th><th>8.05</th><th>35.0</th></tr></table><br><br>
**Обработанные данные:**
<br><table><tr><th>Индексы</th><th>Survived</th><th>Pclass</th><th>Sex</th><th>Fare</th><th>Age</th></tr><tr><th>1</th><th>0</th><th>3</th><th>1</th><th>7.25</th><th>22.0</th></tr><tr><th>2</th><th>1</th><th>1</th><th>0</th><th>71.2833</th><th>38.0</th></tr><tr><th>3</th><th>1</th><th>3</th><th>0</th><th>7.925</th><th>26.0</th></tr><tr><th>4</th><th>1</th><th>1</th><th>0</th><th>53.1</th><th>35.0</th></tr><tr><th>5</th><th>0</th><th>3</th><th>1</th><th>8.05</th><th>35.0</th></tr></table><br><br>
**Ответ:**<br>
При построении классификационного дерева решения на данном датасете важны параметры:<br>Пол (Sex), вес = 0.30051221095823943<br>Класс билета (Pclass), вес = 0.14551470967144137<br><br>