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



GOD_step=0
in_out_neurons = 1
hidden_neurons = 10
n_prev = 100
main_batch = n_prev*10
epoch=100
test_size=0.2


def _load_data(data, n_prev ):
    """
    data should be pd.DataFrame()
    """
    len_data=len(data)-n_prev-GOD_step+1
    X, Y = [], []
    for i in range(len_data):
        X.append(data.iloc[i:i+n_prev].as_matrix())
        Y.append(data.iloc[i+n_prev+GOD_step-1].as_matrix())
        
    reX = np.array(X).reshape(len_data, n_prev, 1)
    reY = np.array(Y).reshape(len_data, -1)

    return reX, reY

def train_test_split(df, test_size, n_prev):
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


(X_train, y_train), (X_test, y_test) = train_test_split(df[["X"]], test_size,n_prev )
#dataframe=pd.Dataframe(X_train,columns("x"))

length_of_sequence = X_train.shape[1]
in_out_neurons = 1

model = Sequential()
#model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
#model.GRU(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(SimpleRNN(hidden_neurons,batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))

model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam")
#early_stoppingをcallbacksで定義　→　validationの誤差値(val_loss)の変化が収束したと判定された場合に自動で終了
#modeをauto　→　収束の判定を自動で行う．
#patience　→　判定値からpatienceの値の分だけのepoch学習. 変化がなければ終了
#patience=0　→　val_lossが上昇した瞬間終了
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
callback=CSVLogger("./data_out/RNN_history.csv")
model.fit(X_train, y_train, batch_size=main_batch, nb_epoch=epoch, validation_split=0.05)

predicted = model.predict(X_test)
dataf =  pd.DataFrame(predicted[:])
dataf.columns = ["OUTPUT_RNN"]
dataf["OUTPUT_ORI"] = y_test[:]
print(dataf)
#plt.figure()
dataf.plot()
plt.show()