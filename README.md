# Autoencoder
Задача: Сделайть модель для очистки документов от шума и “грязи”.

<a name="4"></a>
## [Оглавление:](#4)
1. [Callbacks](#1)
2. [Формирование параметров загрузки](#2)
3. [Создание сети](#3)

Импортируем нужные библиотеки.
```
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
import pandas as pd                                             # Загружаем библиотеку Pandas
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
import os                                                       # Импортируем модуль os для загрузки данных
import time                                                     # Импортируем модуль time
from google.colab import drive                                  # Подключим гугл диск
from tensorflow.keras.models import Model                       # Загружаем абстрактный класс базовой модели сети от кераса
# Подключим необходимые слои
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, \
                                    concatenate, Activation, MaxPooling2D, Conv2D, \
                                    BatchNormalization, Dropout, MaxPooling1D, UpSampling2D
from tensorflow.keras import backend as K                       # Подключим бэкэнд Керас
from tensorflow.keras.optimizers import Adam                    # Подключим оптимизатор
from tensorflow.keras import utils                              # Подключим utils
from tensorflow.keras.utils import plot_model                   # Подключим plot_model для отрисовки модели
from tensorflow.keras.preprocessing import image                # Подключим image для работы с изображениями
from PIL import Image                                           # Подключим Image для работы с изображениями
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback # 
import tensorflow as tf                                         # Импортируем tensorflow
import random                                                   # Импортируем библиотеку random
```
Объявим необходимые функции.
```
def plotImages(xTrain, pred, shape=(420, 540, 3)):
    '''
    Функция для вывода изображений
    '''
    n = 5  # количество картинок, которые хотим показать
    plt.figure(figsize=(18, 6)) # указываем размеры фигуры
    for i in range(n): # для каждой картинки из n(5)
        index = np.random.randint(0, pred.shape[0]) # startIndex - начиная с какого индекса хотим заплотить картинки
        # Показываем картинки из тестового набора
        ax = plt.subplot(2, n, i + 1) # выведем область рисования Axes
        plt.imshow(xTrain[index].reshape(shape)) # отрисуем правильные картинки
        ax.get_xaxis().set_visible(False) # скрываем вывод координатной оси x
        ax.get_yaxis().set_visible(False) # скрываем вывод координатной оси y

        # Показываем восстановленные картинки
        ax = plt.subplot(2, n, i + 1 + n) # выведем область рисования Axes 
        plt.imshow(pred[index].reshape(shape)) # отрисуем обработанные сеткой картинки 
        ax.get_xaxis().set_visible(False) # скрываем вывод координатной оси x
        ax.get_yaxis().set_visible(False) # скрываем вывод координатной оси y
    plt.show()
```
```
def getMSE(x1, x2):
    '''
    Функция среднеквадратичной ошибки
    
    Return:
        возвращаем сумму квадратов разницы, делённую на длину разницы
    '''
    x1 = x1.flatten() # сплющиваем в одномерный вектор
    x2 = x2.flatten() # сплющиваем в одномерный вектор
    delta = x1 - x2 # находим разницу
    return sum(delta ** 2) / len(delta)
```
```
def load_images(images_dir, img_height, img_width): 
    '''
    Функция загрузки изображений, на вход принемает имя папки с изображениями, 
    высоту и ширину к которой будут преобразованы загружаемые изображения

    Return:
        возвращаем numpy массив загруженных избражений
    '''
    list_images = [] # создаем пустой список в который будем загружать изображения
    for img in sorted(os.listdir(images_dir)): # получим список изображений и для каждого изображения
    # добавим в список изображение в виде массива, с заданными размерами
        list_images.append(image.img_to_array(image.load_img(os.path.join(images_dir, img), \
                                                            target_size=(img_height, img_width))))
    return np.array(list_images)
```
[:arrow_up:Оглавление](#4)
<a name="1"></a>
## Callbacks.
```
# Остановит обучение, когда valloss не будет расти
earlyStopCB = EarlyStopping(
                    monitor='loss',
                    min_delta=0,
                    patience=8,
                    verbose=0,
                    mode='min',
                    baseline=None,
                    restore_best_weights=False,
                )
```
```
# Вывод шага обучения
def on_epoch_end(epoch, logs):
  lr = tf.keras.backend.get_value(model.optimizer.learning_rate)
  print(' Коэффициент обучения', lr)

lamCB = LambdaCallback(on_epoch_end=on_epoch_end)
```
```
# Меняет шаг обучения
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=5, min_lr=0.00000001)
```
[:arrow_up:Оглавление](#4)
<a name="2"></a>
## Формируем параметры загрузки данных
```
xLen = 30                           # Анализируем по 30 прошедшим точкам 
valLen = 500                        # Используем 500 записей для проверки

trainLen = data.shape[0] - valLen   # Размер тренировочной выборки

# Делим данные на тренировочную и тестовую выборки 
xTrain, xTest = data[:trainLen], data[trainLen + xLen + 2:]

# Масштабируем данные (отдельно для X и Y), чтобы их легче было скормить сетке
xScaler = MinMaxScaler()
xScaler.fit(xTrain)
xTrain = xScaler.transform(xTrain)
xTest = xScaler.transform(xTest)

# Делаем reshape, т.к. у нас только один столбец по одному значению
yTrain, yTest = np.reshape(data[:trainLen, 0], (-1, 1)), np.reshape(data[trainLen + xLen + 2:, 0], (-1, 1)) 
yScaler = MinMaxScaler()
yScaler.fit(yTrain)
yTrain = yScaler.transform(yTrain)
yTest = yScaler.transform(yTest)

# Создаем генератор для обучения
trainDataGen = TimeseriesGenerator(xTrain, yTrain,           # В качестве параметров наши выборки
                               length = xLen, stride = 10,   # Для каждой точки (из промежутка длины xLen)
                               batch_size = 20)              # Размер batch, который будем скармливать модели

# Создаем аналогичный генератор для валидации при обучении
testDataGen = TimeseriesGenerator(xTest, yTest,
                               length = xLen, stride = 10,
                               batch_size = 20)

# Создадим генератор проверочной выборки, из которой потом вытащим xVal, yVal для проверки
DataGen = TimeseriesGenerator(xTest, yTest,
                               length = 30, stride = 10,
                               batch_size = len(xTest))     # Размер batch будет равен длине нашей выборки
xVal = []
yVal = []
for i in DataGen:
    xVal.append(i[0])
    yVal.append(i[1])

xVal = np.array(xVal)
yVal = np.array(yVal)
```
[:arrow_up:Оглавление](#4)
<a name="3"></a>
## Создаем сеть.
```
dataInput = Input(shape = (trainDataGen[0][0].shape[1], trainDataGen[0][0].shape[2]))

Conv1DWay1 = Conv1D(20, 5, activation = 'relu')(dataInput)
Conv1DWay1 = MaxPooling1D(padding = 'same')(Conv1DWay1)

Conv1DWay2 = Conv1D(20, 5, activation = 'relu')(dataInput)
Conv1DWay2 = MaxPooling1D(padding = 'same')(Conv1DWay2)

x1 = Flatten()(Conv1DWay1)
x2 = Flatten()(Conv1DWay2)

finWay = concatenate([x1, x2])
finWay = Dense(200, activation = 'linear')(finWay)
finWay = Dropout(0.15)(finWay)
finWay = Dense(1, activation = 'linear')(finWay)

modelX = Model(dataInput, finWay)
```
Компилируем, запускаем обучение.
```
history = modelX.fit(trainDataGen, 
                    epochs = 15, 
                    verbose = 1,
                    validation_data = testDataGen)
```
Выведем график обучения.
```
plt.plot(history.history['loss'], 
         label = 'Точность на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label = 'Точность на проверочном наборе')
plt.ylabel('Средняя ошибка')
plt.legend()
plt.show()
```
График автокорреляции.
```
currModel = modelX
predVal, yValUnscaled = getPred(currModel, xVal[0], yVal[0], yScaler)
showPredict(0, 500, 0, predVal, yValUnscaled)
showCorr([0], 11, predVal, yValUnscaled)
```
[:arrow_up:Оглавление](#4)

[Ноутбук](https://colab.research.google.com/drive/1F1XRgISbk0EaB-eZc-5kAXFfqTArxeAd?usp=sharing)
