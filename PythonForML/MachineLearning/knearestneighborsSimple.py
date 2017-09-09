from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import random
import pandas as pd

style.use('fivethirtyeight')

# there are two group in our dummy set = k and r
#dataset = {'k':[[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6]]}
#new_feat = [5,6]

#[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_feat[0], new_feat[1])
#plt.show()

def k_nearest_neighbors(data, predict, k=3):
    # we always want more k neighbors to be more than the data
    # otherwise, we have k=2 and classes are 3, 
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')

    # how to compare the data points - we have to compare with all the other
    # data points - or we can use the radius to find out which points are closer
    distances = []
    # iterate over all the training data - one by one
    for group in data:
        # for each train data feature
        for features in data[group]:
            # find the distance between the point and all the other data point
            e_dis = np.linalg.norm(np.array(features) - np.array(predict))

            # store the distance and group
            distances.append([e_dis, group])    # list of list

    # simply sort the data with distance - get k points
    # get the sorted distance of the k neighbors - create a list of group
    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))     #find the most common group
    
    # get the most common groups in the nearest neighbors i.e. the vote result
    # get ['group', 'how many are voting for it] - use the how many to get the conf
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k    # how many of the k's was in majority
    return vote_result, confidence

"""
result = k_nearest_neighbors(dataset, new_feat, k=3)
print(result)

# for all the dataset group, plot the points both x, y with size and color
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feat[0], new_feat[1], s=200, color=result)
plt.show()
"""

accuracies = []

for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))] # 80 percent of data
    test_data = full_data[-int(test_size*len(full_data)):]  # last 20 % of data

    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            else:
                print(vote, confidence)
            total += 1


    accuracies.append(correct/total)
    #print('Accuracy: ', correct/total)


print(sum(accuracies)/len(accuracies))










