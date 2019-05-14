import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import GRU
import matplotlib.pyplot as plt




def seikei_data(df, GOD_step,n_prev):
    
    X, Y = [], []
    for i in range(0, len(df)-n_prev, GOD_step):
        X.append(df.iloc[i:i+n_prev].as_matrix())
        Y.append(df.iloc[i+n_prev+GOD_step-1].as_matrix())
    X_out=np.array(X)
    Y_out=np.array(Y)
    return X_out, Y_out

def create_dataset(df, train_size, GOD_step,n_prev):
    pos = round(len(df) * (train_size))#round→切り捨て
    target_column=1
    
    
    trainX, trainY = seikei_data(df.iloc[0:pos], GOD_step,n_prev)
    testX, testY   = seikei_data(df.iloc[pos:], GOD_step,n_prev)
    return trainX, trainY, testX, testY

    
def rmsle(y_pred, y_true):
    return np.sqrt(np.square(np.log(y_true + 1) - np.log(y_pred + 1)).mean())


def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std

df = pd.read_csv('./data/international-airline-passengers.csv',
                    usecols=[1],
                    engine='python',
                    skipfooter = 3)


#print(dataset)

steps_of_history = 1
steps_in_future = 1
train_size=0.7
GOD_step=1
n_prev=10
in_out_neurons = 1
hidden_neurons = 300

trainX, trainY, testX, testY = create_dataset(df,train_size,GOD_step,n_prev)


print(trainX)
print(trainY)
print(trainX.shape)


#最適化手法sgd, rsmprop, adagrad, adadelta, adam, adamax, nadam
#RNN（再帰型ニューラルネットワーク）では最適化が遅いのでrmspropを使うといいらしい。
#https://keras.io/ja/layers/recurrent/

length_of_train=len(trainX)
#print(trainX.shape)
#trainX.resize((1,length_of_sequences,length_of_train))
#trainY.resize((1,length_of_sequences,length_of_train))
#print(trainX.shape)
model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, n_prev, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("tanh"))

model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(trainX, trainY, batch_size=1, epochs=15, validation_split=0.001)



