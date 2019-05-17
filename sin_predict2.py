import pandas as pd
import numpy as np
import math
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import matplotlib.pyplot as plt



GOD_step=1#CSVファイルより固定
inout_node = 1
hidden_node = 100
epoch=100 #学習epoch数
test_size=0.2
timesteps= 1
inout_node= 1


def create_dataset(df, test_size):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train = df.iloc[0:ntrn,0].values.reshape(ntrn, -1, 1)
    y_train = df.iloc[0:ntrn,1].values.reshape(ntrn, -1)
    X_test  = df.iloc[ntrn:,0].values.reshape(len(df)-ntrn, -1, 1)
    y_test  = df.iloc[ntrn:,1].values.reshape(len(df)-ntrn, -1)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_test, y_test



#ファイルをpd形式で取ってきて成型
df = pd.read_csv('./data/output_Runge_Lorenz2.csv',
                    engine='python',
                )
df = df.rename(columns={0: 't'})
df = df.rename(columns={1: 't+1'})
X_train, y_train, X_test, y_test = create_dataset(df, test_size )


#(batch_size, timesteps, input_dim)：入力サイズ
#(batch_size, input_dim)：出力サイズ
model = Sequential()
#model.add(GRU(hidden_neurons, batch_input_shape=(None, timesteps, inout_node), return_sequences=False))
#model.add(LSTM(hidden_neurons, batch_input_shape=(None, timesteps, inout_node), return_sequences=False))
model.add(SimpleRNN(hidden_node,batch_input_shape=(None, 1, inout_node), return_sequences=False))
model.add(Dense(inout_node))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, y_train, batch_size=None, nb_epoch=epoch, validation_split=0.05)


y_out_rnn = model.predict(X_test)
dataf =  pd.DataFrame(y_out_rnn[:])
dataf.columns = ["OUTPUT_RNN"]
dataf["OUTPUT_ORI"] = y_test[:]
print(dataf)
dataf.plot()
plt.show()