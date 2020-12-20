import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
raw = read_csv('/Users/mikoriza/Documents/Python/bbri.csv', sep=',', header=0, index_col=0, engine='python', usecols=[0, 4], parse_dates=True)
dataframe = read_csv('/Users/mikoriza/Documents/Python/bbri.csv', sep=',', usecols=[4], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float')

#membuat dataset
def create_dataset(dataset, dimensi):
  dataX, dataY = [], []
  for i in range(len(dataset) - dimensi - 1):
    a = dataset[i:(i + dimensi), 0]
    dataX.append(a)
    dataY.append(dataset[i + dimensi, 0])
  return numpy.array(dataX), numpy.array(dataY)

# print(dataX)
# print(dataY)

# nilai random
numpy.random.seed(7)

# pembagian data train dan data test masing2 sebanyak 50%
train_size = int(len(dataset) * 0.50)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# pembentukan dataset train dan test
dimensi = 10
trainX, trainY = create_dataset(train, dimensi)
testX, testY = create_dataset(test, dimensi)

# model neural network
model = Sequential()
model.add(Dense(10, input_dim=dimensi, activation='relu', use_bias=True, bias_initializer='ones'))  #inisialisasi input layer dan hidden layer
model.add(Dense(10, activation='relu', use_bias=True, bias_initializer='ones'))       #hidden layer 2
model.add(Dense(1, activation='relu'))                                                                  #inisialisasi output layer
model.compile(loss='mean_squared_error', optimizer='adam')                                             #update bobot
model.fit(trainX, trainY, epochs=250, verbose=2)                                                      #fit model

# estimasi performa model
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.3f MSE (%.3f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.3f MSE (%.3f RMSE)' % (testScore, math.sqrt(testScore)))

# hasil prediksi
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# mengubah tipe data hasil prediksi ke integer
trainPredict = trainPredict.astype(int)
testPredict = testPredict.astype(int)

# print hasil prediksi
# print('\nHasil prediksi latihan:\n'+str(trainPredict))
# print('\nHasil tes latihan:\n' +str(testPredict))

# plotting grafik data hasil prediksi saat train
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[dimensi:len(trainPredict)+dimensi, :] = trainPredict

# plotting grafik data hasil prediksi saat test
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(testPredict)+(dimensi*2)+1:len(dataset)-1, :] = testPredict

# menampilkan grafik
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# save model NN
model.save('mandat_backpropagation.h5')