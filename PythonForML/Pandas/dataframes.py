import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

# data frames are like dictionaries
# simple python dictionary - having key and values as list - convert it into the 
# data frame
web_stats = {'Day' : [3, 2, 3, 4, 5],
            'Visitors' : [43, 21, 34, 45, 56],
            'Bounce_Rate' : [65, 76, 76, 65, 34]}

df = pd.DataFrame(web_stats)

# for time series data - the index is that time series data
#print(df)
#print(df.tail())
#print(df.head())
#print(df.tail(2))
#print(df.set_index('Day'))
#print(df.head())

#df = df.set_index('Day')
#df.set_index('Day', inplace=True)

#print(df['Day'].head(2))

#print(df.head())

#print(df.Visitors.tolist())
print(df['Visitors'].tolist())

# dataframes pandas accepts list of list
print(df[['Visitors', 'Bounce_Rate']])
# wont work : print(df[['Visitors', 'Bounce_Rate']].tolist())


# we can simply use numpy array to convert the data frame into a list
print(np.array(df[['Visitors', 'Bounce_Rate']]))

df2 = pd.DataFrame(np.array(df[['Visitors', 'Bounce_Rate']]))

print(df2)
