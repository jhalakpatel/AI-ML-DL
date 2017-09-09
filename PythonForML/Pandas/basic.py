import pandas
import datetime

# pandas work with all kind of data types
# csv, html -conversion - data frame can be used in the same
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 1, 1)

# referencing the yahoo stock API - perform some data analysis on the visualization
df = data.DataReader("XOM", "yahoo", start, end)

# pandas will put the data into data frame - for simple data visualizationa
# work well with massive dataset
print(df.head())
df['Adj Close'].plot()

plt.show()
