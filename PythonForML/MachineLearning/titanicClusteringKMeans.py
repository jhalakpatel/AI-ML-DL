import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from matplotlib import style
style.use('ggplot')

'''
survival
name
sex
age
cabin
lifeboat
destination
body identification number
ticket number
parent
spouse
fare passenger

Need to predict if the passenger will survive or not - using kMeans - 
separate ppl in two groups - live or die based on the data and features
'''
df = pd.read_excel('/Users/jhalakpatel/Desktop/ML/PythonForML/MachineLearning/titanic.xls')
# data preprocessing
# need to translate text data into numerical data
# fill of missing data
df.drop(['body', 'boat', 'sex', 'fare'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values
    for col in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        # for a particular col
        if df[col].dtype != np.int64 and df[col].dtype!=np.float64:
            col_contents = df[col].values.tolist()
            # get all the unique elements in the cols
            unique_elements = set(col_contents)
            x = 0
            # assinn numeric values to each unique element
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            # simply call pandas function map - to map cols to new cols values
            # map the convert_to_int function with values of df cols and convert int lost
            df[col] = list(map(convert_to_int, df[col]))

    return df
df = handle_non_numerical_data(df)
#print(df.head())
# using clustering algo - we can predict the values of new cluster
# clustering is not supervised we can not predict for all the data

# convert data frames to X input set
X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)      # simple training the classifier

# simply get the cluster label for the classifier
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

# concepts - clusters are assigned randomly - of them them can be 0 ==> actual 0
# or 0 ==> actual 1
print(correct/len(X))

print(df.head())
