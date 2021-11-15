# Байесовский классификатор
# Представим, что нам на склад поступили детали, которые были произведены на двух разных станках.
# Известны следующие характеристики поступивших на склад изделий:
# Всего на склад поступило деталей — 10000 шт.
# Будем считать, что нам не известны доли стандартных изделий, производимых на станках №1 и №2, но допустим,
# вместо этого мы знаем средние размеры диаметров изделий и стандартное отклонение диаметра изделий на каждом станке.

# Станок №1 производит детали размером 64(d) мм в диаметре и стандартным отклонением 4(q) мм.
# Станок №2 производит детали размером 52 мм в диаметре и стандартным отклонением в 2 мм.

# Немаловажное условие — вся совокупность изделий описывается нормальным распределением или распределением Гаусса.

# Будем считать, что в процессе приемки продукции произошел небольшой инцидент,
# в результате которого все изделия перемешались.
# Наша задача перебрать детали и для каждой определить вероятность того,
# что она была произведена на станке №1 или на станке №2.
# Также мы будем считать, что деталь произведена на том станке, вероятность которого выше.

import pandas as pd
import math

ALL_DETAILS = 8000
DETAILS = [[6000, 64, 4],  # [N, d, q1], N - кол-во деталей
           [4000, 52, 2]]
FILE_PATH = "Data/"
PB1 = DETAILS[0][0] / ALL_DETAILS
PB2 = DETAILS[1][0] / ALL_DETAILS
FCO = 1 / (DETAILS[0][2] * math.sqrt(math.pi * 2))  # FIRST_CONSTANT_FOR_TYPE_ONE
FCT = 1 / (DETAILS[1][2] * math.sqrt(math.pi * 2))  # FIRST_CONSTANT_FOR_TYPE_TWO

print("Тестирование Байесовского классификатора")
print("Загружаю данные", end="")
detailsTable = pd.read_csv(FILE_PATH + "details.csv", index_col=0)
detailsTable = detailsTable.sample(frac=1).reset_index(drop=True)
del detailsTable['Масса']
detailsTable['Вероятность 1 типа'] = 0
detailsTable['Вероятность 2 типа'] = 0
detailsTable['Прогноз'] = 0

for i in range(ALL_DETAILS):
    print(f"\r Делаю прогноз: Прогресс {(i / ALL_DETAILS):.2%}", end="")
    zn = (detailsTable.loc[i, 'Диаметр'] - DETAILS[0][1]) ** 2
    ch = 2 * DETAILS[0][2] ** 2
    PB1X = FCO * math.exp(-zn / ch)

    zn = (detailsTable.loc[i, 'Диаметр'] - DETAILS[1][1]) ** 2
    ch = 2 * DETAILS[1][2] ** 2
    PB2X = FCT * math.exp(-zn / ch)

    detailsTable.loc[i, 'Вероятность 1 типа'] = PB1 * PB1X / (PB1 * PB1X + PB2 * PB2X)
    detailsTable.loc[i, 'Вероятность 2 типа'] = PB2 * PB2X / (PB1 * PB1X + PB2 * PB2X)

    if detailsTable.loc[i, 'Вероятность 1 типа'] > detailsTable.loc[i, 'Вероятность 2 типа']:
        detailsTable.loc[i, 'Прогноз'] = 1
    elif detailsTable.loc[i, 'Вероятность 1 типа'] < detailsTable.loc[i, 'Вероятность 2 типа']:
        detailsTable.loc[i, 'Прогноз'] = 2

detailsTable['Результат'] = detailsTable['Прогноз'] == detailsTable['Выход']
count = detailsTable[detailsTable['Результат'] == True]['Результат'].count()
print("\rТочность: {:.2%}".format(count / ALL_DETAILS))
