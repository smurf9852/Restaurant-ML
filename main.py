import pandas
from matplotlib import dates
import datetime as dt #default
from datetime import datetime #default
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import weather
import utils
from models import GradientBoosting
from models import FullyConnected

from sklearn.preprocessing import MinMaxScaler

# TO ASK
# what to do with missing days

# Forecasting op maand niveau
# Koppeling maken tussen werknemers namen

# 1 TRY STUFF

# Take last year
# Seperate difference stores
# Take different dates

# 2 GET INFORMATION

# Pre / Post Corona difference
# information gain from features
# Plots of data - general data range
# Sales per weekday
# Bar graph of rollout over 2/4 weeks

# 3 TRY MODELS

# fully connected
# autoregressor - ARIMA
# CNN
# identity identification problem
# gaussian processes for lower-upper boundary
# Klassiek algorithme

# 4 OTHER STUFF

# ! Read papers
# Research Kaggle

# https://www.kaggle.com/c/favorita-grocery-sales-forecasting
# https://www.kaggle.com/aremoto/retail-sales-forecast


# Forecasting on monthly level

def make_validation_set(df):
    pass


bin_size_euro = 250
bin_size = bin_size_euro * 100

df = pandas.read_csv('sales_payments.csv')

df = utils.df_to_datetime_format(df)
df = utils.create_summation(df)
df = utils.df_add_dates(df)
df = utils.df_drop_columns_and_outliers(df)
df = utils.insert_past_revenues(df)
df = utils.binning(df, bin_size)

df_weather = weather.get_weather()

df = weather.merge_weather(df, df_weather)

df = utils.remove_nan_values(df)

date = '2020-01-01'


scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled[df_scaled.columns])
df['store_id'] = df.index.get_level_values('store_id')

df_pre, df_post = utils.split_on_date(df, date)

train_x, train_y, test_x, test_y = utils.create_train_test(df_pre)

model = FullyConnected(train_x.shape[1])
model.model.fit(train_x.astype(float), train_y, validation_data = (test_x.astype(float), test_y), epochs = 10)
output = model.predict(test_x.astype(float))

model.test(test_x.astype(float), test_y)


#1 DAY ROLLOUT

store_id = 3
post_x, post_y = utils.create_train_test(df_post, test = False, store_id = store_id)

outputs = []
targets = []

for i, x in post_x.iloc[0:20].iterrows():
    
    target = post_y.loc[i]   
    
    output = model.model.predict(x.astype(float).to_numpy().reshape(1,-1))
    
    output = int(output.item())

    outputs.append(output//100)
    targets.append(target//100)
    
    
plt.bar(np.arange(len(outputs)) - 0.25, outputs, 0.25, label = 'output 1 day rollout', color = 'dodgerblue')
plt.bar(np.arange(len(outputs)) + 0.25, targets, 0.25, label = 'target', color = 'lightcoral')

plt.legend()
plt.title('1 day vs total rollout for store 3 in January 2020')
plt.xlabel('days since 03 jan 2020')
plt.ylabel('euro')
plt.show()




























