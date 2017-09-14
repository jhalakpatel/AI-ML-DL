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
from nltk.classify import ClassifierI
from statistics import mode 

# class inheriting from ClassifierI
class VoteClassifier(ClassifierI):
	# passing a list of classifiers with *
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		# return the highest frequency of the votes
		return mode(votes)	  

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)

		# get how many classifier are in favor of the mode votes
		choice_votes = votes.count(mode(votes))
		# get the overall confidence of the votes
		confs = choice_votes / len(votes)
		return confs


# perform data preprocessing
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]

# dont shuffle so we know what is pos and what is neg
#random.shuffle(docs)
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
print(len(featuresets))
# positive dataset - testing
#training_set = featuresets[:1900]
#testing_set = featuresets[1900:]

# negative dataset - testing
training_set = featuresets[100:]
testing_set = featuresets[:100]

# Basic NLTK classifier
nltk_clf = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo Accuracy', nltk.classify.accuracy(nltk_clf, testing_set)*100)

# MNB
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print('MNB Algo Accuracy', nltk.classify.accuracy(MNB_clf, testing_set)*100)

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


# get the voting classifier
voted_clf = VoteClassifier(nltk_clf, 
							MNB_clf,  
							BernoulliNB_clf, 
							LogisticRegression_clf, 
							SGDClassifier_clf, 
							LinearSVC_clf, 
							NuSVC_clf)

print('voted classifier : ', voted_clf)
print("voted classifier accuracy percentage:", (nltk.classify.accuracy(voted_clf, testing_set))*100)

print('Classification:', voted_clf.classify(testing_set[0][0]), 'Confidence %:', voted_clf.confidence(testing_set[0][0])*100)
print('Classification:', voted_clf.classify(testing_set[1][0]), 'Confidence %:', voted_clf.confidence(testing_set[1][0])*100)
print('Classification:', voted_clf.classify(testing_set[2][0]), 'Confidence %:', voted_clf.confidence(testing_set[2][0])*100)
print('Classification:', voted_clf.classify(testing_set[3][0]), 'Confidence %:', voted_clf.confidence(testing_set[3][0])*100)
print('Classification:', voted_clf.classify(testing_set[4][0]), 'Confidence %:', voted_clf.confidence(testing_set[4][0])*100)
print('Classification:', voted_clf.classify(testing_set[5][0]), 'Confidence %:', voted_clf.confidence(testing_set[5][0])*100)

# need to find the accuracy on the testing set
# possible acc = 75% - we are 100% acc on neg and 50% on pos