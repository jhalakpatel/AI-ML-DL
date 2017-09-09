""" 
Basics 
1. there is no such thing as training and testing - in k nearest neighbors
2. it can do parallel processing - and thread it and work things in parallel
3. calculate euclidean distance
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
# treating the file as csv file - use pandas to read the file
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# for the missing data - handle the outliers
df.replace('?',-99999, inplace=True)
# remove the useless data - i.e. IDs - its an outlier
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

# create the classifier
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [1, 1, 1, 1, 2, 2, 3, 2, 1]])

# why reshaping is required - if there are multiple samples - then 
# we have use predict for both of them
#example_measures = example_measures.reshape(len(example_measures),-1)
#prediction = clf.predict(example_measures)
#print(prediction)
