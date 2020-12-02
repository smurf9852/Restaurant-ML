import pandas
import datetime
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import dateutil.relativedelta
import seaborn as sns


def df_to_datetime_format(df):
    df['created_at'] = pandas.to_datetime(df['created_at'])
    df['updated_at'] = pandas.to_datetime(df['updated_at'])
    
    return df

def df_add_dates(df):

    df['year'] = [x.year for x in df.index.get_level_values('created_at')]
    df['month'] = [x.month for x in df.index.get_level_values('created_at')]
    df['weeknumber'] = [x.isocalendar()[1] for x in df.index.get_level_values('created_at')]
    df['weekday'] = [x.isocalendar()[2] for x in df.index.get_level_values('created_at')]
        
    return df

def df_drop_columns_and_outliers(df):
    
    df.drop(columns=['id', 'employee_id', 'customer_id','total_amount_without_tax'], inplace = True)
    df.drop(index = df[df['total_amount'] <= 10000].index, inplace = True)
    
    return df

def get_past_revenue(df, row, days = 0, months = 0):
    
    this_day = row.name[0]
    this_store_id = row.name[1]
    
    prev_day = this_day 
    prev_day = prev_day - datetime.timedelta(days = days)
    prev_day = prev_day - dateutil.relativedelta.relativedelta(months= months)
    
    get_previous_day = df.loc[(df.index.get_level_values('created_at') == prev_day) & (df.index.get_level_values('store_id') == this_store_id)]
    
    if not get_previous_day.empty:
        return get_previous_day['total_amount'].item()
    else:
        return np.nan

def bin_f(x):
    if x.time() < datetime.time(13):
        return "00:00-12:59"
    else:
        return "13:00-23:59"
    
def continues_to_bins(series, bin_size):
    
    max_size = series.max() + bin_size
    
    bins = np.arange(0, max_size , bin_size)
    labels = bins + bin_size//2
    
    return pandas.cut(series, bins, labels=labels[:-1])

def replace_nan_values(data):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(data)
    return imp.transform(data)

def remove_nan_values(df):
    return df.dropna()

def create_test_train(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    train_x = train.copy().drop(columns=['total_amount'])
    test_x = test.copy().drop(columns=['total_amount'])
    train_y = train['total_amount']
    test_y = test['total_amount']
    
    return train_x, test_x, train_y, test_y

def create_summation(df, shift = False):
    if shift:
        return df.groupby(by=[df['created_at'].dt.date,'shift','store_id','organisation_id']).sum()
    else:
        return df.groupby(by=[df['created_at'].dt.date,'store_id','organisation_id']).sum()
    
def substract_months(series, months):
    return  [x - dateutil.relativedelta.relativedelta(months=months) for x in series]

def insert_past_revenues(df):
    
    df['revenue_1daysago'] = df.apply(lambda row: get_past_revenue(df, row, 1), axis=1)
    df['revenue_2daysago'] = df.apply(lambda row: get_past_revenue(df, row, 2), axis=1)
    df['revenue_lastweek'] = df.apply(lambda row: get_past_revenue(df, row, 7), axis=1)
    df['revenue_lastmonth'] = df.apply(lambda row: get_past_revenue(df, row, 0, 1), axis=1)
    #df['revenue_lastyear'] = df.apply(lambda row: get_past_revenue(df, row, 0, 12), axis=1)
    
    return df

def binning(df, bin_size = 50000):
    
    df['total_amount'] = continues_to_bins(df['total_amount'], bin_size)
    df['revenue_1daysago'] = continues_to_bins(df['revenue_1daysago'], bin_size)
    df['revenue_2daysago'] = continues_to_bins(df['revenue_2daysago'], bin_size)
    df['revenue_lastweek'] = continues_to_bins(df['revenue_lastweek'], bin_size)
    df['revenue_lastmonth'] = continues_to_bins(df['revenue_lastmonth'], bin_size)
    #df['revenue_lastyear'] = continues_to_bins(df['revenue_lastyear'], bin_size)
    
    return df

def split_corona(df):    
       
    pre_corona = df.iloc[df.index.get_level_values(0) < '2020-03-01']
    post_corona = df.iloc[df.index.get_level_values(0) >= '2020-03-01']
    
    return (pre_corona, post_corona)

def split_on_date(df, date):    
       
    pre = df.iloc[df.index.get_level_values(0) < date]
    post = df.iloc[df.index.get_level_values(0) >= date]
    
    return (pre, post)

def create_train_test(df, test = True, store_id = None):
    
    if store_id:
        df = df.iloc[df.index.get_level_values('store_id') == store_id]
        
    if test:
        
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]
        train_x = train.copy().drop(columns=['total_amount'])
        test_x = test.copy().drop(columns=['total_amount'])
        train_y = train['total_amount']
        test_y = test['total_amount']
        
        return (train_x, train_y, test_x, test_y)
    
    else:
        
        x = df.copy().drop(columns=['total_amount'])
        y = df['total_amount']
        
        return (x,y)
        
def create_subplots_of_columns(df):

    df_ = df.reset_index()
    df_ = df_.loc[df_['store_id'] == 3]
    df_['total_amount'] = df_['total_amount'].astype(int)
    df_[['date', 'total_amount', 'tempC','sunHour', 'month']].plot(x='date', title = 'store 3', subplots=True, figsize=(20,15))
    plt.show()
    
def create_correlation_plot(df):
    df_ = df.reset_index()    
    df_ = convert_to_int(df_, ['total_amount','revenue_1daysago', 'revenue_2daysago', 'revenue_lastweek',
           'revenue_lastmonth', 'revenue_lastyear','store_id'])
    
    corr = df_.drop(columns=['organisation_id']).corr()
    plt.figure(figsize=(10,10))
    plt.title('Correlation Matrix')
    sns.heatmap(corr, 
                cmap = sns.color_palette("rocket_r"),
                annot=True, fmt=".3f",
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )
    plt.show()
    
def convert_to_int(df, column_names):
    
    for column_name in column_names:    
        df[column_name] = df[column_name].astype(int)
    
    return df

def get_past_revenue_array(df, store_id, date_today, number_of_days):
    
    df = df.iloc[df.index.get_level_values('store_id') == store_id]
    df = df.iloc[df.index.get_level_values('date') < date_today]
    df = df.iloc[-number_of_days:]
    
    return df['total_amount'].to_list()
       

