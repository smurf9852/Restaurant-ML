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


bin_size_euro = 500
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

date = '2020-02-01'

scaler = MinMaxScaler()


df_scaled = df.copy()
df_scaled[df_scaled.columns] = scaler.fit_transform(df_scaled[df_scaled.columns])

df_scaled['store_id'] = df_scaled.index.get_level_values('store_id')

df_pre, df_post = utils.split_on_date(df_scaled, date)

train_x, train_y, test_x, test_y = utils.create_train_test(df_pre)

#model = GradientBoosting()
#model.fit(train_x, train_y)
#model.test(test_x, test_y)

model = FullyConnected(train_x.shape[1])

def pd_to_np_float(pd):
    return pd.to_numpy().astype('float32')



train_x, train_y, test_x, test_y = pd_to_np_float(train_x), pd_to_np_float(train_y), pd_to_np_float(test_x), pd_to_np_float(test_y)



model.model.fit(train_x, train_y, epochs = 100, validation_data = (test_x,test_y))






















# store_id = 3

# post_x, post_y = utils.create_train_test(df_post, test = False, store_id = store_id)

# revenue_array = utils.get_past_revenue_array(df, store_id, date, 30)

# outputs = []
# labels = []

# day = datetime.strptime(date, "%Y-%m-%d")

# for i, x in post_x.iloc[0:30].iterrows():
    
#     target = post_y.loc[i]   
    
#     r_1 = revenue_array[-1]
#     r_2 = revenue_array[-2]
#     r_7 = revenue_array[0]
    
#     x.revenue_1daysago = r_1
#     x.revenue_2daysago = r_2
#     x.revenue_lastweek = r_7
    
#     x_vec = model.toNumpyVector(x)
#     output = model.predict(x_vec)
    
#     output = int(output.item())

#     outputs.append(output)
#     revenue_array.append(output)
#     labels.append(day.strftime("%Y-%m-%d"))

#     del revenue_array[0]
#     day = day + dt.timedelta(days=1)
    
#     print(f"{day.strftime('%Y-%m-%d')} - {day.strftime('%A')} - {target/100} - {output/100}")
    
# d = {"data" : outputs, "labels" : labels}





    
    
    
































