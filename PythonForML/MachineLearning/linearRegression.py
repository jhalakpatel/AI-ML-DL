""" Linear Regression """
"""
    1. working on google stock data from quadl
    2. simple linear regression


Thoughts:
    1. what can be a feature and what can be a label
    2. Adj. Close can not be a label
    3. we want some point in future we want to predict - that can be a label
    4. 
"""

# python data analysis library - pandas
import pandas as pd

# to import data from quandl - database for stock and other financial data
import quandl

# basic math
import math

# np - scientific computing library - has arrays - python does not have arrays
# can create N dimensional arrays to work with
import numpy as np 

# for scaling the data - done on features - 
# features should be between -1 to 1 - like standard deviation and mean will come 
# into picture
# svm - to do regression using svm
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

# libaries for graph plotting
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import datetime

# store the classifier locally i.e. trained classifier - or other data or files 
import pickle

# df - data frame - dataframe is pandas primitive to simply get the data
df = quandl.get('WIKI/GOOGL')

# from the data frame - get the required columns are list of cols
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]

# using the interesting data - create relavant data
# create new cols - simple in data frame
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# update data frame using the column index
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# what we want out regression model to predit - here we want to predict - 
# the price - i.e. closing price at the end of the day in future
forecast_col = 'Adj. Close'

# in real world we wont have all the data - we need to treat the outlier properly
df.fillna(-99999, inplace=True)     # replace the NaN data with some value, 

# will round it up to the whole a number of the celing
# will be the number of data out

# we want to predict - 1% in future about the stock prices
forecast_out = int(math.ceil(0.1*len(df)))

# here we are predicting 33 days in the future - trend for google stocks
#print(forecast_out)

# shift the col negatively - each row will be the adjusted time in the future
# for the forecast columns - we have shifted labels 30 days for the training data
df['label'] = df[forecast_col].shift(-forecast_out)

# create input from the data frame - drop lavel
# create the input - except for the labels
X = np.array(df.drop(['label', 'Adj. Close'], 1))
X = preprocessing.scale(X)          # scales X before training - normalization
X_lately = X[-forecast_out:]        # last values - we want to predict the X_lately
X = X[:-forecast_out]               # first values

# remove data frame which does not have data
df.dropna(inplace=True)
y = np.array(df['label'])       # get the labels for current data

#print(len(X), len(y))
# take feature and label, shuffle them up - and split the data into training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)
# using sklearn model_selection library to split the data into train and test data
# fit the classifier - from sklearn library obtain the LinearRegression
#clf = LinearRegression(n_jobs=1)
# clf = svm.SVR(kernel='linear')
# clf = svm.SVR(kernel='poly')

# train the data
#clf.fit(X_train, y_train)

# we should save the classifier after training - we want to avoid retraining the data
#with open('linearregression.pickle', 'wb') as f:
#    pickle.dump(clf, f)

# load the stored classifier
# simple pickle based file operating - where we can store the classifier 
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

# test the data - test on other data - avoid overfitting the data
accuracy = clf.score(X_test, y_test)

# data we want to predict - we do not have labels for them
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan     # NAN data for new cols

# get the last data frame element and name attribute
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# populate with the new date and forecast values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # set the first cols as not a number
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
