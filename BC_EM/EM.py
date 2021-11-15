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
import numpy as np
import math

FILE_PATH = 'Data/'


class EM_Algoritm:
    def __init__(self, trainData: pd.DataFrame, testData: pd.DataFrame = None, noise=False):

        self.accuracy = 0
        self.n = noise

        if self.n:
            print("\033[1m\033[32m\nСоздается объект EM алгоритма")

        if testData is None:
            self.dataPreparation(trainData)
        else:
            self.trainData = trainData
            self.testData = testData

        if self.n:
            print("Инициализирую начальные параметры \033[0m")
        self.trainData['Прогноз'] = np.random.randint(1, 3, self.trainData.shape[0])
        self.countFG = 0
        self.countSG = 0
        self.M()
        self.countAll = self.countFG + self.countSG

    def dataPreparation(self, trainData):
        trainDataset = detailsTable.sample(frac=0.8, random_state=0)
        try:
            del trainDataset['Выход']
        except:
            pass
        testDataset = detailsTable.drop(trainDataset.index)
        self.trainData = trainDataset
        self.testData = testDataset

    def train(self):
        if self.n: print("\nНачинаю обучение")
        while self.delta:
            self.E()
            self.M()
        self.calculatingAccuracy()
        if self.n: print("Обучение выполненно")

    def E(self):
        l = 0.0000000000000001
        verFGC = self.countFG / self.countAll
        verSGC = self.countSG / self.countAll

        self.trainData['Прогноз'] = 0

        for j in range(len(self.trainData.index)):
            if self.n: print(f"\rВыполнено {j / len(self.trainData.index) :.1%}", end="")
            verFG = verFGC
            verSG = verSGC
            for i in self.trainData.columns:
                znach = self.trainData.loc[self.trainData.index[j], i]

                znF = (znach - self.paramFG[i][0]) ** 2
                chF = (2 * (self.paramFG[i][1] ** 2))
                fcF = 1 / (self.paramFG[i][1] * math.sqrt(math.pi * 2) + l)
                verFG *= fcF * math.exp(- znF / chF if chF else l)

                znS = (znach - self.paramSG[i][0]) ** 2
                chS = (2 * (self.paramSG[i][1] ** 2))
                fcS = 1 / (self.paramSG[i][1] * math.sqrt(math.pi * 2) + l)
                verSG *= fcS * math.exp(- znS / chS if chS else l)

            verFG = float(verFG if verFG else l)
            verSG = float(verSG if verSG else l)
            ver1 = verFG / (verFG + verSG)
            ver2 = verSG / (verFG + verSG)
            if ver1 > ver2:
                self.trainData.loc[self.trainData.index[j], 'Прогноз'] = 1
            else:
                self.trainData.loc[self.trainData.index[j], 'Прогноз'] = 2
        if self.n: print("\r \n\n", end="")

    def M(self):
        d1 = self.trainData[self.trainData['Прогноз'] == 1]
        d2 = self.trainData[self.trainData['Прогноз'] == 2]

        if len(d1.index) == 0 or len(d2.index) == 0:
            self.trainData['Прогноз'] = np.random.randint(1, 3, self.trainData.shape[0])
            d1 = self.trainData[self.trainData['Прогноз'] == 1]
            d2 = self.trainData[self.trainData['Прогноз'] == 2]

        self.delta = self.countFG - len(d1.index)

        self.countFG = len(d1.index)
        self.countSG = len(d2.index)

        self.paramFG = {}
        self.paramSG = {}

        for i in d1.columns:
            p11 = d1[i].sum() / self.countFG
            p21 = d2[i].sum() / self.countSG

            p12 = d1[i].std()
            p22 = d2[i].std()

            self.paramFG[i] = [p11, p12]
            self.paramSG[i] = [p21, p22]

        if self.n: print(
            "Разбиение на две группы \n  Группа 1: {} шт \n  Группа 2: {} шт \n  Дельта: {}".format(self.countFG,
                                                                                                    self.countSG,
                                                                                                    self.delta))

    def calculatingAccuracy(self):
        l = 0.0000000000000001
        verFGC = self.countFG / self.countAll
        verSGC = self.countSG / self.countAll

        self.testData['Прогноз'] = 0

        try:
            columns = list(self.testData.columns)
            columns.remove('Выход')
        except ValueError:
            columns = self.testData.columns

        for j in range(len(self.testData.index)):
            if self.n: print(f"\rВыполнено {j / len(self.testData.index) :.1%}", end="")
            verFG = verFGC
            verSG = verSGC
            for i in columns:
                znach = self.testData.loc[self.testData.index[j], i]

                znF = (znach - self.paramFG[i][0]) ** 2
                chF = (2 * (self.paramFG[i][1] ** 2))
                fcF = 1 / (self.paramFG[i][1] * math.sqrt(math.pi * 2) + l)
                verFG *= fcF * math.exp(- znF / chF if chF else l)

                znS = (znach - self.paramSG[i][0]) ** 2
                chS = (2 * (self.paramSG[i][1] ** 2))
                fcS = 1 / (self.paramSG[i][1] * math.sqrt(math.pi * 2) + l)
                verSG *= fcS * math.exp(- znS / chS if chS else l)

            verFG = float(verFG if verFG else l)
            verSG = float(verSG if verSG else l)
            ver1 = verFG / (verFG + verSG)
            ver2 = verSG / (verFG + verSG)
            if ver1 > ver2:
                self.testData.loc[self.testData.index[j], 'Прогноз'] = 1
            else:
                self.testData.loc[self.testData.index[j], 'Прогноз'] = 2

        print(self.testData)
        count = sum(self.testData['Прогноз'] == self.testData['Выход'])
        accuracy = count / len(self.testData.index)
        if accuracy < 0.5:
            accuracy = 1 - accuracy
        if self.n:
            print("\r \n", end="")
            if accuracy < 0.7:
                print(f"\033[1m\033[31m Точность: {accuracy:.1%}")
            elif accuracy < 0.9:
                print(f"\033[1m\033[33m Точность: {accuracy:.1%}")
            else:
                print(f"\033[1m\033[32m Точность: {accuracy:.1%}")
            print('\033[0m')

    def classification(self, data: pd.DataFrame):
        l = 0.0000000000000001
        verFGC = self.countFG / self.countAll
        verSGC = self.countSG / self.countAll

        data['Прогноз'] = 0

        try:
            if not (data.columns == self.trainData.columns).all():
                print()
                print("\033[1m\033[31m Ошибка индексов в таблице:")
                print("\033[1m\033[31m \t Индексы в тренировочной таблице не сходятся с индексами у входной таблицы")
                print()
                return -1
        except ValueError:
            print()
            print("\033[1m\033[31m Ошибка индексов в таблице:")
            print("\033[1m\033[31m \t Индексы в тренировочной таблице не сходятся с индексами у входной таблицы")
            print()
            return -1

        columns = data.columns

        for j in range(len(data.index)):
            if self.n: print(f"\rВыполнено {j / len(data.index) :.1%}", end="")
            verFG = verFGC
            verSG = verSGC
            for i in columns:
                znach = data.loc[data.index[j], i]

                znF = (znach - self.paramFG[i][0]) ** 2
                chF = (2 * (self.paramFG[i][1] ** 2))
                fcF = 1 / (self.paramFG[i][1] * math.sqrt(math.pi * 2) + l)
                verFG *= fcF * math.exp(- znF / chF if chF else l)

                znS = (znach - self.paramSG[i][0]) ** 2
                chS = (2 * (self.paramSG[i][1] ** 2))
                fcS = 1 / (self.paramSG[i][1] * math.sqrt(math.pi * 2) + l)
                verSG *= fcS * math.exp(- znS / chS if chS else l)

            verFG = float(verFG if verFG else l)
            verSG = float(verSG if verSG else l)
            ver1 = verFG / (verFG + verSG)
            ver2 = verSG / (verFG + verSG)
            if ver1 > ver2:
                data.loc[data.index[j], 'Прогноз'] = 1
            else:
                data.loc[data.index[j], 'Прогноз'] = 2
        if self.n:
            print('\033[1m\033[32m\n\nКлассификация прошла успешно! \033[0m')


detailsTable = pd.read_csv(FILE_PATH + "details.csv", index_col=0)
detailsTable = detailsTable.sample(frac=1).reset_index(drop=True)
emA = EM_Algoritm(detailsTable, noise=True)
emA.train()
