#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)]
labels_train = labels_train[:len(labels_train)]

#########################################################
### your code goes here ###

from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0)     # creating a support vector classifier - simple classifier using support vector machines

# training the dataset
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t1 = time()
ans = clf.predict(features_test)
print "inference time:", round(time()-t1, 3), "s"

print "10:" , ans[10]
print "26:" , ans[26]
print "50:" , ans[50]
count = 0
for l in ans:
    if (l==1): 
        count = count + 1

print "1's : ", count

print "Accurracy :", clf.score(features_test, labels_test)
#########################################################


