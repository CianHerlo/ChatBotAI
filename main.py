import numpy as np
import pandas as pd

from subprocess import check_output
from tensorflow.keras.core import Dense, Activation, Dropout
from tensorflow.keras.recurrent import LTSM
from tensorflow.keras.models import Sequential
from sklearn.cross_validation import train_test_split
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis

apple_dataset = pd.read_csv('Apple.csv', header=0)
apple_dataset
