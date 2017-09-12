import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()
'''
polarity
 0 = negative
 2 = neutral
 4 = positive

id
date
query
user
tweet
'''
# read the sentiment or tweeter data, preprocess and store them
# modify the setiment labels
def init_process(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1, 0]
                elif initial_polarity == '4':
                    initial_polarity = [0, 1]

                # last col is the tweet
                tweet = line.split(',')[-1]

                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


# create lexicon from the random sample of the dataset
def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                # for every 2500th sample
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    # append the content with the tweet
                    content += ' '+tweet
                    words = word_tokenize(content)
                    # lemmatize all the tokenized words and store them in a list
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon+words))
                    #print(counter, len(lexicon))
        except Exception as e:
            print(str(e))


    with open('lexicon.pickle','wb') as f:
        pickle.dump(lexicon, f)


def convert_to_vec(fin, fout, lexicon_pickle):
    # open pickle file for reading
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)

    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            # create lemmatized word list with in built for loop
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            # create feature list with 0 values for all the lexicon
            # feature will have the count of the sample words in the lexicon
            features = np.zeros(len(lexicon))
           
            for word in current_words:
                if word.lower() in lexicon:
                    # get the index in lexicon list to which 
                    # the given word  belong
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            print('features',features)
            print(counter)
            features = list(features)
            outline = str(features)+'::'+str(label)+'\n'
            outfile.write(outline)
        print(counter)

def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    # shuffle all the rows of data frame
    # create permutation map or list of all the df rows
    # simple copy df rows with permutation to the new df
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('trained_set_shuffled.csv', index=False)


# vectorize the data into bag of words model - prior to training the data
#  vectorize the data first and then feed it to the network
def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                # eval basically run python code in python shell
                features = list(eval(line.split('::')[0]))
                label = list(eval(line.split('::')[1]))
                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print(counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)

#init_process('trainingandtestdata/training.1600000.processed.noemoticon.csv','train_set.csv')
#init_process('trainingandtestdata/testdata.manual.2009.06.14.csv','test_set.csv')
#create_lexicon('train_set.csv')
convert_to_vec('test_set.csv', 'processed-test-set.csv', 'lexicon.pickle')
#shuffle_data('train_set.csv')
#create_test_data_pickle('processed-test-set.csv')
