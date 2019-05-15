import pandas as pd
import numpy as np
import math
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import RNN
from keras.layers import GRU
import matplotlib.pyplot as plt



GOD_step=50
in_out_neurons = 1
hidden_neurons = 10
length_of_sequences = 100
epoch=100


def _load_data(data, n_prev = 100):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev-GOD_step+1):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev+GOD_step-1].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.2, n_prev = 100):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)


    return (X_train, y_train), (X_test, y_test)



df = pd.read_csv('./data/output_Runge_Lorenz.csv',
#                    usecols=[1],
                    engine='python',
                    header =None
                )

df = df.rename(columns={0: 'X'})
df = df.rename(columns={1: 'Y'})
df = df.rename(columns={2: 'Z'})
#df[["sin_t"]].head(steps_per_cycle * 2).plot()


(X_train, y_train), (X_test, y_test) = train_test_split(df[["X"]], n_prev =length_of_sequences)


model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(X_train, y_train, batch_size=600, nb_epoch=epoch, validation_split=0.05)

predicted = model.predict(X_test)
dataf =  pd.DataFrame(predicted[:200])
dataf.columns = ["predict"]
dataf["input"] = y_test[:200]
print(dataf)
#plt.figure()
dataf.plot()
plt.show()