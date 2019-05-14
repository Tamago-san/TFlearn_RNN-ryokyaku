import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib.pyplot as plt

data = np.random.rand(250,5)
labels = np_utils.to_categorical( (np.sum(data, axis=1) <1)*1 )

print(labels.shape)
print(labels)

print((np.sum(data, axis=1) > 2.5))
print((np.sum(data, axis=1) > 2.5)*1)
print(True*1)
print(False*1)