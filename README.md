# Фич инжиниринг
Задача состоит в том, чтобы посмотреть, получится или нет, добавив дополнительные фичи/признаки, избавиться от автокрреляции.

<a name="4"></a>
## [Оглавление:](#4)
1. [Загрузка данных](#1)
2. [Формирование параметров загрузки](#2)
3. [Создание сети](#3)

Импортируем нужные библиотеки.
```
import pandas as pd                                             # Загружаем библиотеку Pandas
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
%matplotlib inline
from tensorflow.keras.optimizers import Adam, Adamax            # Подключим оптимизаторы
from tensorflow.keras.models import Sequential, Model           # Загружаем абстрактный класс базовой модели сети от кераса
from tensorflow.keras.layers import concatenate, Input, Dense, Dropout, MaxPooling1D    # Подключим необходимые слои
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv1D, LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Номализация данных
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # Генератор данных
import gdown                                                    # Загрузка датасетов из облака google
```
Определим фукции для работы.
```
def getPred(currModel, xVal, yVal, yScaler):
    '''
    Функция рассчитываем результаты прогнозирования сети
    Args:
        В аргументы принимает сеть (currModel) и проверочную выборку
    Return:
        Выдаёт результаты предсказания predVal
        И правильные ответы в исходной размерности yValUnscaled (какими они были до нормирования)
    '''
    # Предсказываем ответ сети по проверочной выборке
    # И возвращаем исходны масштаб данных, до нормализации
    predVal = yScaler.inverse_transform(currModel.predict(xVal))
    yValUnscaled = yScaler.inverse_transform(yVal)
    
    return predVal, yValUnscaled
```
```
def showPredict(start, step, channel, predVal, yValUnscaled):
    '''
    Функция визуализирует графики, что предсказала сеть и какие были правильные ответы
    Args:
        start - точка с которой начинаем отрисовку графика
        step - длина графика, которую отрисовываем
        channel - какой канал отрисовываем
    '''
    plt.plot(predVal[start:start + step, 0],
            label = 'Прогноз')
    plt.plot(yValUnscaled[start:start + step, channel], 
            label = 'Базовый ряд')
    plt.xlabel('Время')
    plt.ylabel('Значение Close')
    plt.legend()
    plt.show()
```
```
def correlate(a, b):
    '''
    Функция расёта корреляции дух одномерных векторов
    '''
    # Рассчитываем основные показатели
    ma = a.mean() # Среднее значение первого вектора
    mb = b.mean() # Среднее значение второго вектора
    mab = (a * b).mean() # Среднее значение произведения векторов
    sa = a.std() # Среднеквадратичное отклонение первого вектора
    sb = b.std() # Среднеквадратичное отклонение второго вектора
    
    # Рассчитываем корреляцию
    val = 1
    if ((sa > 0) & (sb > 0)):
        val = (mab - ma * mb)/(sa * sb)
    return val
```
```
def showCorr(channels, corrSteps, predVal, yValUnscaled):
    '''
    Функция рисует корреляцию прогнозированного сигнала с правильным,
    cмещая на различное количество шагов назад, для проверки появления эффекта автокорреляции
    Args:
        channels - по каким каналам отображать корреляцию
        corrSteps - на какое количество шагов смещать сигнал назад для рассчёта корреляции
    '''
    # Проходим по всем каналам
    for ch in channels:
        corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
        yLen = yValUnscaled.shape[0] # Запоминаем размер проверочной выборки

    # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
        # Получаем сигнал, смещённый на i шагов назад
        # predVal[i:, ch]
        # Сравниваем его с верными ответами, без смещения назад
        # yValUnscaled[:yLen-i,ch]
        # Рассчитываем их корреляцию и добавляем в лист
        corr.append(correlate(yValUnscaled[:yLen - i, ch], predVal[i:, 0]))

    own_corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно

    # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
        # Получаем сигнал, смещённый на i шагов назад
        # predVal[i:, ch]
        # Сравниваем его с верными ответами, без смещения назад
        # yValUnscaled[:yLen-i,ch]
        # Рассчитываем их корреляцию и добавляем в лист
        own_corr.append(correlate(yValUnscaled[:yLen - i, ch], yValUnscaled[i:, ch]))

    # Отображаем график коррелций для данного шага
    plt.plot(corr, label = 'Предсказание на ' + str(ch + 1) + ' шаг')
    plt.plot(own_corr, label = 'Эталон')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()
```
[:arrow_up:Оглавление](#4)
<a name="1"></a>
## Загрузим данные для анализа.
```
url = 'https://drive.google.com/uc?id=1_dfvJSMa9cQMgnE7jMthQ7CCzR7_Oj9B'
gdown.download(url, None, quiet=False)
```
```
data = pd.read_excel('/content/USD_011994_122020.xlsx')
data.head(3)
```
Получим дополнительные столбцы.
```
data['year'] = pd.DatetimeIndex(data['data']).year
data['month'] = pd.DatetimeIndex(data['data']).month
data['day'] = pd.DatetimeIndex(data['data']).day
```
Удалим не нужные столбцы.
```
data = data.drop(columns = ['nominal', 'data', 'cdx'])
data.head(3)
```
Для анализа возьмем не всю базу, только после 1998 года.
```
data = data[data['year'] > 1998]
```
Дополнительные данные:
- Скользящие средние SMA
- Bollinger bands
```
for sma_period in range(1, 36):
    indicator_name = 'SMA_%d' % (sma_period)
    data[indicator_name] = data['curs'].rolling(sma_period).mean()
```
```
for j in range(10, 20):
    for i in [1, 2, 3, 4]:
        indicator_name1 = 'BB_Up_%d_%d' % (j, i)
        indicator_name2 = 'BB_Down_%d_%d' % (j, i)
        data[indicator_name1] = data['curs'].rolling(j).mean() + i * data['curs'].rolling(j).std()
        data[indicator_name2] = data['curs'].rolling(j).mean() - i * data['curs'].rolling(j).std()
```
Удаляем строки с NaN, преобразуем в массив numpy.
```
data = data.dropna()
data = np.array(data)
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

[Ноутбук](https://colab.research.google.com/drive/1-YSwg3N_snmKxM7pGICV0ph6ffhhrSGq#scrollTo=i0mHUpXZk7Qj&uniqifier=2)
