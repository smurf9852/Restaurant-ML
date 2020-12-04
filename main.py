import pandas
from matplotlib import dates
import datetime as dt #default
from datetime import datetime #default
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

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
#df = utils.binning(df, bin_size)

df_weather = weather.get_weather()

df = weather.merge_weather(df, df_weather)

df = utils.remove_nan_values(df)


def getAverageValue(l):
    
    med = np.nanmedian(l)
    leftovers = [x for x in l if med*1.3 > x and med*0.7 < x and x != np.nan]
    
    if leftovers != []:
        return int(np.mean(leftovers))
    else:
        return np.nan
    

def getSumOfRevenueInMonth(df, timestamp, store):
    
    query = df.loc[(df.index.get_level_values('date') <= timestamp + dt.timedelta(days=15)) & (timestamp - dt.timedelta(days=15) <= df.index.get_level_values('date')) & (df.index.get_level_values('store_id') == store)]
    return query['total_amount'].sum()


def computeGrowthRatio(revenuePrev, revenueNow):
    
    if revenuePrev == 0 or revenueNow == 0:
        return None
    return revenueNow / revenuePrev

for i,row in df.iterrows():
    
    lastYearMinTwoWeek = utils.get_past_revenue(df, row, days = 14, months = 12, datename = 'date')
    lastYearMinOneWeek = utils.get_past_revenue(df, row, days = 7, months = 12, datename = 'date')
    lastYear = utils.get_past_revenue(df, row, days = 0, months = 12, datename = 'date')
    lastYearPlusTwoWeek = utils.get_past_revenue(df, row, days = -14, months = 12, datename = 'date')
    lastYearPlusOneWeek = utils.get_past_revenue(df, row, days = -7, months = 12, datename = 'date')
    
    resultsLastYear = [lastYearMinTwoWeek, lastYearMinOneWeek, lastYear, lastYearPlusOneWeek, lastYearPlusTwoWeek]
    
    averageRevenue = getAverageValue(resultsLastYear)
    
    this_day = row.name[0]
    this_store_id = row.name[1]
    
    sumThisMonth = getSumOfRevenueInMonth(df, this_day, this_store_id)
    sumMonthLastYear = getSumOfRevenueInMonth(df, this_day - dt.timedelta(days=365), this_store_id)
    
    ratio = computeGrowthRatio(sumMonthLastYear, sumThisMonth)
       
    if not math.isnan(averageRevenue) and ratio:
        output = int(ratio * averageRevenue)
        print(f"{this_day.strftime('%d/%m/%Y')} - real {row.total_amount} - output {output} - {(output-row.total_amount)/row.total_amount} - ratio {ratio}")


#ratio is incorrect?
    
    
    
    







#def substract_year(timestamp):
  #  day_last_year = timestamp - dt.timedelta(days=365)
    
    
    















