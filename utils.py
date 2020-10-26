import pandas
import datetime
import numpy as np
from sklearn.impute import SimpleImputer



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

def get_past_revenue(df, row, number_days_back):
    
    this_day = row.name[0]
    this_store_id = row.name[1]
    
    prev_day = this_day - datetime.timedelta(days=number_days_back)
    
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
    

def insert_past_revenues_and_bin(df, bin_size = 50000):
    
    df['revenue_1daysago'] = df.apply(lambda row: get_past_revenue(df, row, 1), axis=1)
    df['revenue_2daysago'] = df.apply(lambda row: get_past_revenue(df, row, 2), axis=1)
    df['revenue_lastweek'] = df.apply(lambda row: get_past_revenue(df, row, 7), axis=1)
    #df['revenue_lastyear'] = df.apply(lambda row: get_past_revenue(df, row, 365), axis=1)
    
    
    df['total_amount'] = continues_to_bins(df['total_amount'], bin_size)
    df['revenue_1daysago'] = continues_to_bins(df['revenue_1daysago'], bin_size)
    df['revenue_2daysago'] = continues_to_bins(df['revenue_2daysago'], bin_size)
    df['revenue_lastweek'] = continues_to_bins(df['revenue_lastweek'], bin_size)
    #df['revenue_lastyear'] = continues_to_bins(df['revenue_lastyear'], bin_size)
    
    return df

def split_corona(df):    
       
    pre_corona = df.iloc[df.index.get_level_values(0) < '2020-03-01']
    post_corona = df.iloc[df.index.get_level_values(0) >= '2020-03-01']
    
    return (pre_corona, post_corona)

def create_train_test(df):
    
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    train_x = train.copy().drop(columns=['total_amount'])
    test_x = test.copy().drop(columns=['total_amount'])
    train_y = train['total_amount']
    test_y = test['total_amount']
    
    return (train_x, train_y, test_x, test_y)