import pandas
import utils

def get_weather():

    df1 = pandas.read_csv('weather/amsterdam 2012 - 2019.csv')
    df2 = pandas.read_csv('weather/amsterdam 2020.csv')\
    
    df = pandas.concat((df1,df2))
    df['date'] = pandas.to_datetime(df['date_time'])
    
    df_noon = df[df['date'].dt.hour == 12]
    
    df_selection = df_noon[['date', 'tempC', 'sunHour']]
    
    df_selection['date'] = [x.to_pydatetime().date() for x in df_selection['date']]
    
    df['tempC'] = utils.continues_to_bins(df['tempC'], 3)
    df['sunHour'] = utils.continues_to_bins(df['sunHour'], 2)
    
    return df_selection
  
def merge_weather(df, df_weather):

    df['copy_index'] = df.index
    df = df.merge(df_weather, how = 'left', left_on = 'created_at', right_on = 'date')
    df.index = pandas.MultiIndex.from_tuples(df['copy_index'], names=('date', 'store_id', 'organisation_id'))
    df.drop(columns = ['copy_index', 'date'], inplace = True)

    return df
    

