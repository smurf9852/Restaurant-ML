import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates



df = pandas.read_csv('sales_payments.csv')


def df_convert(df):
    df['created_at'] = pandas.to_datetime(df['created_at'])
    df['updated_at'] = pandas.to_datetime(df['updated_at'])
    return df
    
df_ = df_convert(df)

times = [x.time().hour for x in df_['created_at']]

#times_objects = dates.date2num(times)

plt.hist(times, bins = 16)
plt.show()



#df_grouped = df.groupby(by=[df['created_at'].dt.date,'store_id','organisation_id']).sum()

print(df.head())

