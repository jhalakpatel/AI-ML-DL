import nltk
import random
import pickle
from nltk.tokenize import word_tokenize
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


doc_f = open('pickled_algos/documents.pickle', 'rb')
docs = pickle.load(doc_f)
doc_f.close()

# store or pickle word features
word_features5k_f = open('pickled_algos/word_features5k.pickle', 'rb')
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

# find the features
def find_features(doc):
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


features_f = open('pickled_algos/featuresets.pickle', 'rb')
featuresets = pickle.load(features_f)
features_f.close()

random.shuffle(featuresets)
# total 10664 in the featuresets
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

open_file = open('pickled_algos/originalnaivebayes5k.pickle', 'rb')
nltk_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/MNB_clf5k.pickle', 'rb')
MNB_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/BernoulliNB_clf5kpickle', 'rb')
BernoulliNB_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LogisticRegression_clf5k.pickle', 'rb')
LogisticRegression_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/SGDClassifier_clf5k.pickle', 'rb')
SGDClassifier_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/SVC_clf5k.pickle', 'rb')
SVC_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/LinearSVC_clf5k.pickle', 'rb')
LinearSVC_clf = pickle.load(open_file)
open_file.close()

open_file = open('pickled_algos/NuSVC_clf5k.pickle', 'rb')
NuSVC_clf = pickle.load(open_file)
open_file.close()

# get the voting classifier - can not pickle voted classifier -
# pickle is used to store python objects - 
# pickle will serialize and deserialize pyhton object while storing them
# like a ojbect or dict - needs to be converted into byte array i.e. serializing
# once the object is stored - they needs to be deserialized
voted_classifier = VoteClassifier(nltk_clf, 
							MNB_clf,  
							BernoulliNB_clf, 
							LogisticRegression_clf, 
							SGDClassifier_clf, 
							LinearSVC_clf, 
							NuSVC_clf)

# simple API wrapper to do the prediction
def sentiment(text):
	features = find_features(text)
	return voted_classifier.classify(features), voted_classifier.confidence(features)