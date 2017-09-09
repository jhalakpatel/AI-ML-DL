# pandas as python data analysis library
import pandas as pd

# simple read a CSV file
df = pd.read_csv('ZILLOW-Z77006_ZRIMFRR.csv')

# create the index for the current CSV
df.set_index('Date', inplace=True)

# save the CSV with updated index
df.to_csv('newcsv.csv')

# print the data frame
print(df.head())

df = pd.read_csv('newcsv.csv', index_col=0)

print(df.head())

# renaming the cols - index is not a column any more
df.columns = ['Austin_HPI']
print(df.head())

df.to_csv('newcsv2.csv')
df.to_csv('newcsv4.csv', header=False)

# pandas can be used to conver the data from one format to other format
df = pd.read_csv('newcsv.csv', names=['Date', 'Austin_HPI'], index_col=0)

# pandas are used to access the data base - put them into the graph or create 
#HTML or JSON format data

df.to_html('example.html')

df = pd.read_csv('newcsv4.csv', names=['Date', 'Austin_HPI'])

print(df.head())

df.rename(columns={'Austin_HPI' : '77006_HPI'}, inplace=True)

print(df.head())
