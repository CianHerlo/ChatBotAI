import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

Apple = 'Apple'
currency = 'Euro'

start = dt.datetime(2017, 1, 1)
end = dt.datetime.now()
data = pd.read_csv('Apple.csv', header=0)