from setting import *
import tensorflow.keras as keras
from LIB.write_readme import add_readme
from LIB.plot import plotImage
from LIB.draw import plot_model
from tensorflow.keras.callbacks import TensorBoard
from keras import models
from keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

LOSS = 'sparse_categorical_crossentropy'
OPTIMIZER = 'adam'
METRICS = ['accuracy']
EPOCHS = 10
DIR = 'CNN/'
TENSORBOARD_LOG = DIR + "/Log/fit/"


def fitModel(name, data, model, loss, optimizer, metrics, log):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    log_dir = log + name
    dot_img_file = f'{log_dir}/model.png'
    plot_model(model, dot_img_file, show_shapes=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x=data['train']['x'],
              y=data['train']['y'],
              epochs=EPOCHS,
              validation_data=(data['test']['x'], data['test']['y']),
              callbacks=[tensorboard_callback])


def build_model(n, kol_layer):
    model = models.Sequential([layers.Flatten(input_shape=(28, 28))])
    for i in range(kol_layer):
        model.add(layers.Dense(n, activation='relu'))
        model.add(layers.Dropout(0.2))

    model.add(layers.Dense(10, activation='softmax'))
    return model


answer = "# Основы CNN на примере MNIST"
answer = add_readme("Задание", {"Текст": "Распознайте рукописные цифры"}, answer)
answer = add_readme("Входные данные", {"Текст": "- x_train и x_test — тренировочный и тестовый набор изображений в "
                                                "оттенках серого и размером (28,28). Всего тренировочных "
                                                "изображений 60000, а тестовых — 10000\n"
                                                "- y_train и y_test — соответствующие"
                                                " метки классов (от 0 до 9)"}, answer)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
plotImage(DIR + "Data/img1.png", x_train[0, :, :])
answer = add_readme("Первая картинка", {"Картинка": "Data/img1.png"}, answer)

x_train, x_test = x_train / 255.0, x_test / 255.0
data = {
    "train": {
        "x": x_train,
        "y": y_train
    },

    "test": {
        "x": x_test,
        "y": y_test
    }
}
layer_kol = [1]
my_model = [264]
for l in layer_kol:
    for n in my_model:
        fitModel(f'default_{n}_{l}', data, build_model(n, l), LOSS, OPTIMIZER, METRICS, TENSORBOARD_LOG)

with open(DIR + "README.md", "w") as readme:
    readme.write(answer)
