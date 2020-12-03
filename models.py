from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, Input, BatchNormalization
from keras.models import Model, Sequential
from sklearn.preprocessing import normalize
import pickle
import datetime
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
# from lightgbm import LGBMRegressor
from sklearn import base
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

class GradientBoosting():
    def __init__(self):
        
        self.model = GradientBoostingClassifier(verbose = 1)
        
    def fit(self, train_x, train_y):
    
        self.model.fit(train_x, train_y.to_numpy())
        
    def test(self, test_x, test_y, echo = True):
    
        output = self.predict(test_x)
        target = test_y.to_numpy()
    
        self.diff = target - output
        self.abs_diff = np.abs(self.diff)
        
        self.mae = np.mean(self.abs_diff)
        self.mae_std = np.std(self.abs_diff)
        
        perfect = np.sum(self.abs_diff == 0.)
        inperfect = len(self.abs_diff) - perfect
        
        if echo:
            print(f"MAE: euro {self.mae/100}, MAE std: euro {self.mae_std/100}, {perfect} spot-on, {inperfect} off")
            
    def predict(self, x):
        return self.model.predict(x)
    
    def toNumpyVector(self, x):
        return x.to_numpy().reshape(1, -1)
    
        
    def graph(self, bin_size = 25000):
        
        plt.title('Distribution of Model output error in Euro')
        plt.xlabel('euro')
        plt.ylabel('dist')
        plt.hist(self.diff/100, bins = np.arange(-5000,5000,bin_size//100))
        plt.xticks([-5000,-2000,-600,0,600,2000,5000])
        plt.show()

class FullyConnected():
    
    def __init__(self, num_feat):
    
        model = Sequential([
            Dense(128, activation='relu', input_shape=(num_feat,)),
            Dropout(.2),
            Dense(32, activation='relu', input_shape=(num_feat,)),
            Dropout(.2),
            Dense(1, activation='relu')
        ])
        model.compile(optimizer='adam', loss='mse',
                      metrics = 
                      [ tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanAbsolutePercentageError()
                          ])
        self.model = model
        
    def fit(self, train_x, train_y, validation_data, epochs = 1):
        self.model.fit(train_x.to_numpy(), train_y.to_numpy(), epochs = epochs, validation_data = validation_data)
    
    def predict(self, x):
        return self.model.predict(x.to_numpy())
    
    def scale(self, df):
        
        df_x = df.copy().drop(columns=['total_amount'])
        df_y = df['total_amount']
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        
        df_x[df_x.columns] = scaler_x.fit_transform(df_x[df_x.columns])
        df_y = scaler_y.fit_transform(df_y.reshape(-1, 1))
        
        return pd.concat([df_x, df_y], axis=1), scaler_x, scaler_y
        
        
        
        
        
        
    

    