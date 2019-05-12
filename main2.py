import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import tflearn
from tflearn.data_utils import load_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split

#データの前処理
def pre_processing(x):

    x = x.iloc[:, 3:13].values

    label_encoder_x_1 = LabelEncoder()
    x[:, 1] = label_encoder_x_1.fit_transform(x[:, 1])

    label_encoder_x_2 = LabelEncoder()
    x[:, 2] = label_encoder_x_2.fit_transform(x[:, 2])

    onehotencoder = OneHotEncoder(categorical_features = [1])
    x = onehotencoder.fit_transform(x).toarray()

    x = x[:, 1:]

    return x

#スケール調整
def scaling(x):
    sc = StandardScaler()
    x = sc.fit_transform(x)

    return x

#ネットワーク構築
def init_network():
    net = tflearn.input_data(shape=[None, 11])
    net = tflearn.fully_connected(net, 6, activation='relu')
    net = tflearn.fully_connected(net, 6, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

    return net

x, y = load_csv('./data/Churn_Modelling.csv', target_column=13, categorical_labels=True, n_classes=2)

x = pd.DataFrame(x, dtype='float')
x = pre_processing(x)
x = scaling(x)

#ニューラルネットワークの初期化
net = init_network()
#モデル構築
model = tflearn.DNN(net, tensorboard_verbose=2, tensorboard_dir='log')
#トレーニング開始
model.fit(x, y, n_epoch=100, batch_size=10, show_metric=True, validation_set=0.2)


#tensorboard --logdir=./log