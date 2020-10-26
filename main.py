import pandas
from matplotlib import dates
import datetime
import numpy as np


import weather
import utils
from models import GradientBoosting

#- Veranderingen gemakkelijk kunnen worden doorgevoerd
#- Makkelijk variabelen toevoegen of verwijderen
#- Gemakkelijk kan worden getest - resultaten in 
#   - Forecast voor 3 random weken?
#   - Afgelopen week
#   - MAE, etc.
#
# van 500 euro bins naar 250 euro bins
# Klassieke methode testen
# Weer toevoegen
# Testen tegen werkelijkheid
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
df = utils.insert_past_revenues_and_bin(df, bin_size = bin_size)

df_weather = weather.get_weather()

df = weather.merge_weather(df, df_weather)

#df, df_post = split_corona(df)

df = utils.remove_nan_values(df)
train_x, train_y, test_x, test_y = utils.create_train_test(df)

model = GradientBoosting()
model.fit(train_x, train_y)
model.test(test_x, test_y)
model.graph()



























