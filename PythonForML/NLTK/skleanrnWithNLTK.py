import nltk
import random
import pickle
from nltk.corpus import movie_reviews
# imort different types of classifier
# type of classifier - naive_bayes, linear regression based, and svm based
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(doc):
    words = set(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in docs]
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

# Basic NLTK classifier
nltk_clf = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo Accuracy', nltk.classify.accuracy(nltk_clf, testing_set)*100)

# MNB
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print('MNB Algo Accuracy', nltk.classify.accuracy(MNB_clf, testing_set)*100)

# GaussianNB
#GaussianNB_clf = SklearnClassifier(GaussianNB())
#GaussianNB_clf.train(training_set)
#print('MNB Algo Accuracy', nltk.classify.accuracy(GaussianNB_clf, testing_set)*100)

# BernoulliNB
BernoulliNB_clf = SklearnClassifier(BernoulliNB())
BernoulliNB_clf.train(training_set)
print('BernoulliNB Algo Accuracy', nltk.classify.accuracy(BernoulliNB_clf, testing_set)*100)


# LogisticRegression
LogisticRegression_clf = SklearnClassifier(LogisticRegression())
LogisticRegression_clf.train(training_set)
print('LogisticRegression Algo Accuracy', nltk.classify.accuracy(LogisticRegression_clf, testing_set)*100)


# SGDClassifier
SGDClassifier_clf = SklearnClassifier(SGDClassifier())
SGDClassifier_clf.train(training_set)
print('SGDClassifier Algo Accuracy', nltk.classify.accuracy(SGDClassifier_clf, testing_set)*100)


# SVC
SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print('SVC Algo Accuracy', nltk.classify.accuracy(SVC_clf, testing_set)*100)


# LinearSVC
LinearSVC_clf = SklearnClassifier(LinearSVC())
LinearSVC_clf.train(training_set)
print('LinearSVC Algo Accuracy', nltk.classify.accuracy(LinearSVC_clf, testing_set)*100)


# NuSVC
NuSVC_clf = SklearnClassifier(NuSVC())
NuSVC_clf.train(training_set)
print('NuSVC Algo Accuracy', nltk.classify.accuracy(NuSVC_clf, testing_set)*100)
