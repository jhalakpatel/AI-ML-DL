'''
    classify - positive or negative connotations
    like divide the emails into spam or not spam
    we can do sentiment analysis - with only two outcomes - pos or neg senitment
'''
import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# create bag of words - with positive and negative category
# will be used for testing and training - these are the features
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories() 
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)
# search of positive and negative word - itsnaive algorithm

# add all the word to the list
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# words with highest frequency are useless
# get frequency distribution of words
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))
#print(all_words["stupid"])
#print(all_words[:10])
# for text classification

# all word keys frequency distribution is not sorted
# we just need to get top 3000 words
word_features = list(all_words.keys())[:3000]

#print(word_features[:10])

def find_features(doc):
    words = set(doc)
    #print('words in doc', words)
    features = {}
    # for each features i.e. 3000 most common words 
    # if the features in the current word list, set the 
    # feature dictionary for the word = true
    for w in word_features:
        # set features[w] = true if, w in words
        features[w] = (w in words)
    
    return features

#print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

# find feature and category - 
# converting the document -- (rev, category) --> convert it into features and category
# each doc ==> list of words and category
# for rev, category in doc --> lookup for the feature word list --> all the features
# add the features for the word list with the category
featuresets = [(find_features(rev), category) for (rev, category) in docs]

# simply divide the featuresets into testing and training sets
training_set = featuresets[:1900]

# for testing, we feed through and get the category and compare with the actual or expected category
testing_set = featuresets[1900:]

# naive bayes based calculation - like a spam classifier
#  Prob. (of pos / review) = Prob (of review/pos) * Prob (of pos) / Prob review over all
# posterior = prior occurrence of x likihood / evidence

# create the classifier
# clf = nltk.NaiveBayesClassifier.train(training_set)

clf_f = open('naivebayes.pickle', 'rb')
clf = pickle.load(clf_f)
clf_f.close()

print('Naive Bayes Algo Accuracy Percent', nltk.classify.accuracy(clf, testing_set)*100)

#save_clf = open('naivebayes.pickle', 'wb')
#pickle.dump(clf, save_clf)
#save_clf.close()
clf.show_most_informative_features(15)
