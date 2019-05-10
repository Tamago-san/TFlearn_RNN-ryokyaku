import pandas as pd
import numpy as np
import tflearn
import matplotlib.pyplot as plt

df = pd.read_csv('./data/international-airline-passengers.csv',
                    usecols=[1],
                    engine='python',
                    skipfooter = 3)

dataset = df.values
dataset = dataset.astype("float32")