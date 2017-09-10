""" basic word to vec example """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange

import tensorflow as tf

### STEP 1 ####
# download the data
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    """ Download file is not present and perform basic validation on the file"""
    if not os.path.exists(filename):
        filename, _ = urllib.reques.urlretrieve(url+filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
                'Failed to verify ' + filename)
    return filename

# global variable, will call the function to download the file
filename = maybe_download('text8.zip', 31344016)

# read the data into lists of strings
def read_data(filename):
    """ Extract the first file enclosed in a zip file as list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

# simply read the first file of the zipped folder - as list of words
vocabulary = read_data(filename)

print('Data size', len(vocabulary))

#### STEP 2 #####
# build the dictionary and replace the rare words with UNK token
vocabulary_size = 50000

def build_dataset(words, n_words):
    """Process raw inputs into a dataset. """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0   # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


data, count, dictionary, reverse_dictionary = build_dataset(vocabulary, vocabulary_size)

del vocabulary
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

##### STEP 3 ####
# function to generate training batch for skip gram model
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1), dtype=np.int32)
    span = 2 * skip_window + 1 # [skip windows on left target skip window on right ]
    buffer = collections.deque(maxlen=span) # simple queue with maxsize = span
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index+span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window
        target_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target = random.randint(0, span - 1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        if data_index == len(data):
            buffer[:] = data[:span]
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    
    # backtrack a little to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i,0], reverse_dictionary[labels[i,0]])

## STEP 4 ###
# build and train skip-gram model

batch_size = 128
embedding_size = 128        # dimensions of the embedding vector
skip_window = 1             # how many words on the left and right
num_skips = 2               # how nany times to reuse an input to generate label

# we pick a random validation set to sample nearest neighbors. here we limit
# the validation samples to words that have low numeric ID, which by construction
# as also most frequent
valid_size = 16 # random set of words to evaluate similarity on
valid_window = 100 # only pick dev sample in the head of distribution
valid_example = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # number of negative examples to sample

graph = tf.Graph()

with graph.as_default():

    # create placeholders for input dataset
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size,1])
    valid_dataset = tf.constant(valid_example, dtype=tf.int32)


    with tf.device('/cpu:0'):
        # look up embeddings for inputs
        # create a vector for embedding mapping
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # perform a lookup in embeddings for training inputs
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    
        # construct variables for NCE loss - negative softmax sampling
        # choose random negative samples, and make sure that they are minimized
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # compute teh average NCE loss for the batch
    # tfe.nce_loss automatically draws a new sample of negative labels 
    # each time we evaluate the loss
    loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    # construct a SGD optimizer using learning rate of 1.0
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # compute cosine similarity between minibatch examples and all embeddings
    # cosine sim is better than L2 distances
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings/norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)


    # Add variable initializer
    init = tf.global_variables_initializer()


## STEP 5 ###
## begin training
num_steps = 100001

with tf.Session(graph=graph) as session:
    # we must initialize all the variables before we use it
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)

        # create simple dictionary to feed the values - with keys and values
        feed_dict = {train_inputs:batch_inputs, train_labels:batch_labels}

        # work on the list of optimizer and loss function , use feed dict 
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # average loss is estimate of the loss over the last 2000 batches
            print('Average loss at step ', step, ': ', average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_example[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s:' % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

#### STEP 6 ###
# visualize the embeddings 

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'more labels than embeddings'
    plt.figure(figsize=(18,18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    xy=(x,y),
                    xytext=(5,2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
        plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib and scipy to show embeddings.')

        
