import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
from matplotlib import style
style.use('ggplot')

def handle_non_numerical_data(df):
    cols = df.columns.values
    for col in cols:
        # for every column, dictionary to hold the mappings
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        # check for non numerical data
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            col_cont = df[col].values.tolist()
            u_elements = set(col_cont)
            x = 0
            for u in u_elements:
                if u not in text_digit_vals:
                   text_digit_vals[u] = x
                   x+=1
            
            # after populating the dictionary - simply map the data frame
            # cols to conversion function
            df[col] = list(map(convert_to_int, df[col]))
    return df
# Populate dataframe from the xls data and do some preprocessing
df = pd.read_excel('titanic.xls')
df_org = pd.DataFrame.copy(df)      # need to copy, can not simply use reference
df.drop(['body', 'boat'], 1, inplace=True)
df.apply(pd.to_numeric, errors='ignore')
df.fillna(0, inplace=True)
df = handle_non_numerical_data(df)


# create input data arrays from the data frame
X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

# create classifier - Hierarchical clustering classifier - does not required numbber
# of cluster as input, can give radius or bandwidth as input
clf = MeanShift()
clf.fit(X)
labels = clf.labels_
cluster_centers = clf.cluster_centers_

print('unique labels : ', len(np.unique(labels)))
print('cluster centers : ', len(cluster_centers))

n_clusters = len(np.unique(labels))
df_org['cluster_group'] = np.nan

for i in range(len(X)):
    # reference index of the data frame i.e. row of the data frame
    # store the cluster group for each row with the label for that row
    # labels are right in order as they are not shuffled
    df_org['cluster_group'].iloc[i] = labels[i]

# with meanshift , we dont know how many cluster it is going to classify into
surv_rates = {}

# for all the cluster centers - may be more than 2 
# as we are doing hierarchical clustering 
for i in range(n_clusters):
    # for each frame where cluster group is "i" store it in tempdf
    temp_df = df_org[ (df_org['cluster_group']==float(i)) ]
    # for each temp df row for cluster group, check if survived is 1
    surv_cluster = temp_df[ (temp_df['survived']==1) ]
    # get surv rate for the given cluster
    surv_rate = len(surv_cluster)/len(temp_df)
    surv_rates[i] = surv_rate

print(df_org.head())
print(surv_rates)
