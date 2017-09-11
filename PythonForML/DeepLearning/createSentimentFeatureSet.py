import nltk
import numpy as np
import random
import pickle
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# word_tokenize will tokenize all the words in the sentence
# i.e. break the words into the array of tokens
# i pulled the chair -- [i, pulled, the, chair]
# steming - will convert the syntatically similar words or synonyms - removes
# 'ing', 'ed' 'er' extra from the word end
lemmatizer = WordNetLemmatizer()
num_lines = 10000000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos,neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            for l in contents[:num_lines]:
                all_words = word_tokenize(l.lower())
                # simply store the list of words
                lexicon += list(all_words) 

    # lemmatize the words
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # get dictionary of the word counts
    # w_count = {'the':21313, 'of':324234, ..}
    
    l2 = []

    # we dont care about super common words like 'the', 'of' as no particual info
    # also we dont want rare words
    for w in w_counts:
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    # return final lexicon
    return l2
        

def sample_handling(sample, lexicon, classification):
    featureset = []
    """
    [
        [[0 1 0 1 1 0], [0 1]] - for features , neg sample class
        [[1 0 1 0 2 0], [1 0]]- for features, pos sample class
    ]
    """
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:num_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            # iterate through all the tokenized and lemmatized words of
            # the sample line
            for word in current_words:
                # if the word exists in the lexicon list
                if word.lower in lexicon:
                    # find the index of the word using .index function
                    index_value = lexicon.index(word.lower)
                    # increament the features for the current index
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    print('Lexicon Length:', lexicon)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    # if dont shuffle - we will have trained the network with all positves weights 
    # first and then with negative weights - we want the data to be more random
    random.shuffle(features)

    # final questions - thus one hot structure will work
    # tf.argmax([3234, 3244]) == tf.argmax([1, 0])

    features = np.array(features)
    testing_size = int(test_size*len(features))

    # in features - [features, label]
    # with list '0' we will get true features    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x, train_y, test_x, test_y

# main function is used so that we can reference this in our script
if __name__ == '__main__':    
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)


"""
All the words have been seen before in the dict:
lexicon array : [chair, table, spoon , telivision]
New sentence comes up : I pulled the chair up to the table
np.zeros(len(lexicon))
[0 0 0 0]
[1 0 0 0]
[1 1 0 0] - after consuming the sentence 
you can create lexicon for all the sentences
"""
