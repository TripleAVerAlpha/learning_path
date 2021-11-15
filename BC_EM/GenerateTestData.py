import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# На завод пришло детали после двух станков
# Детали различаются по диаметру и весу:
# Станок №1 производит детали размером 64 мм (d) в диаметре и стандартным отклонением 4 мм (q1)
#                              и весом в 12 г.(m) и стандартным отклонением в 1 г.(q2)
# Станок №2 производит детали размером 52 мм в диаметре и стандартным отклонением в 2 мм
#                              и весом в 10 г. и стандартным отклонением 0.8 г.
DETAILS = [[7000, 64, 3.5, 14, 2],  # [N, d, q1, m, q2], N - кол-во деталей
           [1000, 52, 2, 9.5, 0.7]]
FILE_PATH = "Data/"

def plotColumn(data, nameColumn):
    columnTO = data[data["Выход"] == 1]
    columnTO = columnTO[nameColumn]
    columnTT = data[data["Выход"] == 2]
    columnTT = columnTT[nameColumn]
    columnTTL = np.concatenate((columnTT, np.random.normal(0, 0, DETAILS[0][0] - DETAILS[1][0])), axis=0)
    dataT = np.transpose(np.array([np.array(columnTO), columnTTL]))

    fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    ax2.hist(dataT, bins=np.arange(columnTT.min() + 1, dataT.max()), label=('Тип №1', 'Тип №2'))

    ax2.legend(loc=(0.65, 0.8))
    ax2.set_title("Распределение параметра: " + nameColumn)
    ax2.yaxis.tick_right()

    plt.savefig(FILE_PATH + nameColumn + ".png")


def getDetails():
    columns = ["Выход", "Диаметр", "Масса"]
    typeOne = []
    for i in range(DETAILS[0][0]):
        typeOne.append(1)

    typeTwo = []
    for i in range(DETAILS[1][0]):
        typeTwo.append(2)

    detailsTypeOne = np.transpose([np.array(typeOne),
                                   np.random.normal(DETAILS[0][1], DETAILS[0][2], DETAILS[0][0]),
                                   np.random.normal(DETAILS[0][3], DETAILS[0][4], DETAILS[0][0])])
    detailsTypeTwo = np.transpose([np.array(typeTwo),
                                   np.random.normal(DETAILS[1][1], DETAILS[1][2], DETAILS[1][0]),
                                   np.random.normal(DETAILS[1][3], DETAILS[1][4], DETAILS[1][0])])
    allDetails = np.vstack((detailsTypeOne, detailsTypeTwo))
    allDetailsTable = pd.DataFrame(columns=columns, data=allDetails)
    return allDetailsTable


print('Готовлю детали \n' + ("-"*50))
details = getDetails()
print(details)
print(("-"*50))
print('Сохраняю \n' + ("-"*50))
details.to_csv(FILE_PATH + "details.csv", sep=',')
print('Строю распределение по Диаметр \n' + ("-"*50))
plotColumn(details, 'Диаметр')
print('Строю распределение по Масса \n' + ("-"*50))
plotColumn(details, 'Масса')
plta = details.plot(x="Масса", y="Диаметр", kind="scatter")
plt.savefig(FILE_PATH + "allParam.png")