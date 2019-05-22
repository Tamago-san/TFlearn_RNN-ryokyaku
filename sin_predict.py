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



GOD_step=5
in_out_neurons = 1
hidden_neurons = 100
n_prev = 50
epoch=1000
test_size=0.2


def _load_data(data, n_prev,fbatch ):
    """
    data should be pd.DataFrame()
    """
    
    X, Y = [], []
    for i in range(0,fbatch*n_prev,n_prev):
        X.append(data.iloc[i:i+n_prev].as_matrix())
#        Y.append(data.iloc[i+n_prev+GOD_step-1].as_matrix())
        Y.append(data.iloc[i+GOD_step:i+n_prev+GOD_step].as_matrix())
    print(np.array(X).shape)
    reX = np.array(X).reshape(fbatch, n_prev, 1)
    reY = np.array(Y).reshape(fbatch, n_prev, 1)

    return reX, reY

def train_test_split(df, test_size, n_prev):
    """
    This just splits data to training and testing parts
    """
    allsize_batch=int((len(df)-GOD_step)/n_prev)
    train_batch=int(allsize_batch*(1 - test_size))
    test_batch =allsize_batch-train_batch
    print(len(df))
    print(allsize_batch)
    print(train_batch)
    print(test_batch)
    X_train, y_train = _load_data(df.iloc[0:int(n_prev*train_batch)+GOD_step], n_prev,train_batch)
    X_test, y_test = _load_data(df.iloc[int(n_prev*train_batch):], n_prev,test_batch)


    return (X_train, y_train), (X_test, y_test) ,train_batch,test_batch




df = pd.read_csv('./data/output_Runge_Lorenz.csv',
#                    usecols=[1],
                    engine='python',
                    header =None
                )

df = df.rename(columns={0: 'X'})
df = df.rename(columns={1: 'Y'})
df = df.rename(columns={2: 'Z'})
#df[["sin_t"]].head(steps_per_cycle * 2).plot()


(X_train, y_train), (X_test, y_test),train_batch,test_batch = train_test_split(df[["X"]], test_size,n_prev )
#dataframe=pd.Dataframe(X_train,columns("x"))

length_of_sequence = X_train.shape[1]
in_out_neurons = 1

model = Sequential()
#model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons),
                    return_sequences=True,
                    kernel_initializer='random_normal'))
#model.add(SimpleRNN(hidden_neurons,batch_input_shape=(None, length_of_sequence, in_out_neurons)
#                    ,return_sequences=True
#                    ,kernel_initializer='random_normal'))
model.add(Dense(in_out_neurons,kernel_initializer='random_normal'))
model.add(Activation("tanh"))
model.compile(loss="mean_squared_error", optimizer="adam")
#early_stoppingをcallbacksで定義　→　validationの誤差値(val_loss)の変化が収束したと判定された場合に自動で終了
#modeをauto　→　収束の判定を自動で行う．
#patience　→　判定値からpatienceの値の分だけのepoch学習. 変化がなければ終了
#patience=0　→　val_lossが上昇した瞬間終了
#early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
csv_logger=CSVLogger("./data_out/RNN_history.dat", separator='\t')
history=model.fit(X_train, y_train, batch_size=train_batch, nb_epoch=epoch, validation_split=0.,verbose=1,callbacks=[csv_logger])
predicted = model.predict(X_test)
#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
print(history)
dataf =  pd.DataFrame(predicted.flatten())
dataf.columns = ["OUTPUT_RNN"]
dataf["OUTPUT_ORI"] = y_test.flatten()
#print(dataf)
#plt.figure()
#dataf.plot()
#plt.show()
loss = pd.read_table('./data_out/RNN_history.dat',
#                    usecols=[1],
                    engine='python'
                )
                
plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.title("test (Keras.ver)")
p1 = plt.plot(dataf["OUTPUT_RNN"], label="label-A")
p2 = plt.plot(dataf["OUTPUT_ORI"], label="label-B")

plt.xlabel("step")
#plt.ylabel("predo")
plt.legend(["OUTPUT_RNN", "OUTPUT_ORI"],
#           ["wa~i!", "sugo~i!"],
           fontsize=20,
           loc=1,
 #          title="LABEL NAME",
           prop={'size':6})


plt.subplot(122)
plt.title("loss (Keras.ver)")
p1 = plt.plot(loss["loss"], label="label-A")

plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["loss"],
#           ["wa~i!", "sugo~i!"],
           fontsize=20,
           loc=1,
#           title="LABEL NAME",
           prop={'size':6})

plt.show()