import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
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
pos = open('pos.txt', 'r', encoding='utf-8', errors='replace').read()
neg = open('neg.txt', 'r', encoding='utf-8', errors='replace').read()

all_words = []
docs = []
# j is adjective, r is adverb and v is verb
# allowed_word_types = ["J", "R", "V"]
allowed_word_types = ["J"]

for r in pos.split('\n'):
	docs.append( (r, 'pos') )
	words = word_tokenize(r)
	postag = nltk.pos_tag(words)
	for w in postag:
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

for r in neg.split('\n'):
	docs.append( (r, 'neg') )
	words = word_tokenize(r)
	negtag = nltk.pos_tag(words)
	for w in negtag:
		# check the part of speech - pos tagging - part of speech tagging
		if w[1][0] in allowed_word_types:
			all_words.append(w[0].lower())

# save the document in the pickle file
save_docs = open('pickled_algos/documents.pickle', 'wb')
pickle.dump(docs, save_docs)
save_docs.close()

# store the frequency distribution of the words
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

# store or pickle word features
save_word_features = open('pickled_algos/word_features5k.pickle', 'wb')
pickle.dump(word_features, save_word_features)
save_word_features.close()

# find the features
def find_features(doc):
	# tokenize the words in the doc for iteration
    words = word_tokenize(doc)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev), category) for (rev, category) in docs]

features_f = open('pickled_algos/featuresets.pickle', 'wb')
pickle.dump(featuresets, features_f)
features_f.close()

random.shuffle(featuresets)

print(len(featuresets))
# total 10664 in the featuresets
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# Basic NLTK classifier
nltk_clf = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Algo Accuracy', nltk.classify.accuracy(nltk_clf, testing_set)*100)
nltk_clf.show_most_informative_features(15)

save_clf = open('pickled_algos/originalnaivebayes5k.pickle', 'wb')
pickle.dump(nltk_clf, save_clf)
save_clf.close()

# MNB
MNB_clf = SklearnClassifier(MultinomialNB())
MNB_clf.train(training_set)
print('MNB Algo Accuracy', nltk.classify.accuracy(MNB_clf, testing_set)*100)

save_clf = open('pickled_algos/MNB_clf5k.pickle', 'wb')
pickle.dump(MNB_clf, save_clf)
save_clf.close()

# BernoulliNB
BernoulliNB_clf = SklearnClassifier(BernoulliNB())
BernoulliNB_clf.train(training_set)
print('BernoulliNB Algo Accuracy', nltk.classify.accuracy(BernoulliNB_clf, testing_set)*100)

save_clf = open('pickled_algos/BernoulliNB_clf5kpickle', 'wb')
pickle.dump(BernoulliNB_clf, save_clf)
save_clf.close()

# LogisticRegression
LogisticRegression_clf = SklearnClassifier(LogisticRegression())
LogisticRegression_clf.train(training_set)
print('LogisticRegression Algo Accuracy', nltk.classify.accuracy(LogisticRegression_clf, testing_set)*100)

save_clf = open('pickled_algos/LogisticRegression_clf5k.pickle', 'wb')
pickle.dump(LogisticRegression_clf, save_clf)
save_clf.close()

# SGDClassifier
SGDClassifier_clf = SklearnClassifier(SGDClassifier())
SGDClassifier_clf.train(training_set)
print('SGDClassifier Algo Accuracy', nltk.classify.accuracy(SGDClassifier_clf, testing_set)*100)

save_clf = open('pickled_algos/SGDClassifier_clf5k.pickle', 'wb')
pickle.dump(SGDClassifier_clf, save_clf)
save_clf.close()

# SVC
SVC_clf = SklearnClassifier(SVC())
SVC_clf.train(training_set)
print('SVC Algo Accuracy', nltk.classify.accuracy(SVC_clf, testing_set)*100)

save_clf = open('pickled_algos/SVC_clf5k.pickle', 'wb')
pickle.dump(SVC_clf, save_clf)
save_clf.close()

# LinearSVC
LinearSVC_clf = SklearnClassifier(LinearSVC())
LinearSVC_clf.train(training_set)
print('LinearSVC Algo Accuracy', nltk.classify.accuracy(LinearSVC_clf, testing_set)*100)

save_clf = open('pickled_algos/LinearSVC_clf5k.pickle', 'wb')
pickle.dump(LinearSVC_clf, save_clf)
save_clf.close()

# NuSVC
NuSVC_clf = SklearnClassifier(NuSVC())
NuSVC_clf.train(training_set)
print('NuSVC Algo Accuracy', nltk.classify.accuracy(NuSVC_clf, testing_set)*100)

save_clf = open('pickled_algos/NuSVC_clf5k.pickle', 'wb')
pickle.dump(NuSVC_clf, save_clf)
save_clf.close()

# get the voting classifier - can not pickle voted classifier -
# pickle is used to store python objects - 
# pickle will serialize and deserialize pyhton object while storing them
# like a ojbect or dict - needs to be converted into byte array i.e. serializing
# once the object is stored - they needs to be deserialized
voted_clf = VoteClassifier(nltk_clf, 
							MNB_clf,  
							BernoulliNB_clf, 
							LogisticRegression_clf, 
							SGDClassifier_clf, 
							LinearSVC_clf, 
							NuSVC_clf)


print("voted classifier accuracy percentage:", (nltk.classify.accuracy(voted_clf, testing_set))*100)

# simple API wrapper to do the prediction
def sentiment(text):
	features = find_features(text)
	return voted_classifier.classify(features), voted_classifier.confidence(features)