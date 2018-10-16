import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset) - look_back):
		a = dataset[i:(i + look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return dataX, dataY


dfin = pd.read_csv('D:/MobileTrafficPrediction/data/processed/smsin_embedding.csv')
dfout = pd.read_csv('D:/MobileTrafficPrediction/data/processed/smsout_embedding.csv')
clus = pd.read_csv('D:/MobileTrafficPrediction/data/processed/clusterDTW.csv')
clus = OneHotEncoder(sparse=False).fit_transform(clus['clusterID'].reshape((-1, 1)))
dfcallin = pd.read_csv('D:/MobileTrafficPrediction/data/processed/callin_embedding.csv')
dfcallout = pd.read_csv('D:/MobileTrafficPrediction/data/processed/callout_embedding.csv')

train_walking, test_walking = [], []

train_smsin, test_smsin = [], []
train_smsout, test_smsout = [], []
train_clus, test_clus = [], []
train_callin, test_callin = [], []
train_callout, test_callout = [], []
trainX, trainY, testX, testY = [], [], [], []


look_back = 6
for i in range(1, 1000):
    #df1 = pd.read_csv('E:/Italy/gridTraffic30/grid' + str(i) + '.csv')
    df1 = pd.read_csv('D:/MobileTrafficPrediction/data/processed/gridTraffic/gridTraffic/grid'+str(i)+'.csv')
    df1 = df1.rename(columns={0: 'time', 1: 'sms_in', 2: 'sms_out', 3: 'call_in', 4: 'call_out', 5: "traffic"})
    data = df1.iloc[:, -1].values
    dataTrain = data[:int(len(data)/3 * 2)]
    dataTest = data[int(len(data)/3 * 2):]
    tmpX, tmpY = create_dataset(dataTrain, look_back)
    trainX = trainX + tmpX
    trainY = trainY + tmpY
    select = dfin.iloc[i-1]
    select1 = dfout.iloc[i-1]
    select2 = clus[i-1]
    select3 = dfcallin.iloc[i-1]
    select4 = dfcallout.iloc[i-1]
    for k in range(len(tmpX)):
        train_smsin.append(select)
        train_smsout.append(select1)
        train_clus.append(select2)
        train_callin.append(select3)
        train_callout.append(select4)
    tmpX1, tmpY1 = create_dataset(dataTest, look_back)
    testX = testX + tmpX1
    testY = testY + tmpY1
    for k in range(len(tmpX1)):
        test_smsin.append(select)
        test_smsout.append(select1)
        test_clus.append(select2)
        test_callin.append(select3)
        test_callout.append(select4)


print(numpy.array(trainX).shape)
print(numpy.array(testX).shape)
print(numpy.array(trainY).shape)
print(numpy.array(testY).shape)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
trainX = scaler.fit_transform(numpy.array(trainX))
testX = scaler.transform(testX)
scaler1 = MinMaxScaler(feature_range=(0, 1))
trainY = scaler1.fit_transform(numpy.array(trainY).reshape(-1, 1))
testY = scaler1.transform(numpy.array(testY).reshape(-1, 1))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, 1))
testX = numpy.reshape(testX, (testX.shape[0], look_back, 1))
train_smsin = numpy.reshape(train_smsin, (len(train_smsin), 6))
test_smsin = numpy.reshape(test_smsin, (len(test_smsin), 6))
train_smsout = numpy.reshape(train_smsout, (len(train_smsout), 6))
test_smsout = numpy.reshape(test_smsout, (len(test_smsout), 6))
train_clus = numpy.reshape(train_clus, (len(train_clus), 5))
test_clus = numpy.reshape(test_clus, (len(test_clus), 5))
train_callin = numpy.reshape(train_callin, (len(train_callin), 6))
test_callin = numpy.reshape(test_callin, (len(test_callin), 6))
train_callout = numpy.reshape(train_callout, (len(train_callout), 6))
test_callout = numpy.reshape(test_callout, (len(test_callout), 6))
print(trainX.shape)
print(testX.shape)
print('finish loading')

# create and fit the LSTM network
"""
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
"""
in1 = Input(shape=(look_back, 1))
lstm_out = LSTM(32)(in1)
in2 = Input(shape=(6,))
in3 = Input(shape=(6,))
in4 = Input(shape=(5,))
in5 = Input(shape=(6,))
in6 = Input(shape=(6,))
x = keras.layers.concatenate([lstm_out, in2])
x = keras.layers.concatenate([x, in3])
x = keras.layers.concatenate([x, in4])
x = keras.layers.concatenate([x, in5])
x = keras.layers.concatenate([x, in6])
#x = Dense(6, activation='relu', name = 'dense1')(lstm_out)
#y = Dense(1, activation='sigmoid', name = 'dense2')(lstm_out)
y = Dense(1, activation='sigmoid', name = 'dense2')(x)
model = Model(inputs=[in1,in2,in3,in4,in5,in6], outputs=y)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit([trainX,train_smsin,train_smsout,train_clus,train_callin,train_callout], trainY, epochs=100, batch_size=100, verbose=2)


trainPredict = model.predict([trainX, train_smsin, train_smsout, train_clus, train_callin,train_callout])
testPredict = model.predict([testX, test_smsin, test_smsout, test_clus, test_callin, test_callout])

# invert predictions
trainPredict = scaler1.inverse_transform(trainPredict)
trainY = scaler1.inverse_transform(trainY)
testPredict = scaler1.inverse_transform(testPredict)
testY = scaler1.inverse_transform(testY)

trainScore = math.sqrt(mean_squared_error(trainY[:], trainPredict[:, 0:1]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:], testPredict[:, 0:1]))
print('Test Score: %.2f RMSE' % (testScore))




'''
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset), :] = testPredict

# plot baseline and predictions
plt.figure(figsize = (20,15))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
'''

#N = 1000
#smsin
#Train Score: 10.69 RMSE
#Test Score: 10.63 RMSE
#smsout
#Train Score: 10.74 RMSE
#Test Score: 10.61 RMSE
#smsin+smsout
#Train Score: 10.67 RMSE
#Test Score: 10.63 RMSE
#smsin+smsout+cluster
#Train Score: 10.63 RMSE
#Test Score: 10.54 RMSE
#sms+call+cluster
#Train Score: 10.56 RMSE
#Test Score: 10.51 RMSE