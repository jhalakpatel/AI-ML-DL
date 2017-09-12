import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# variables
n_nodes_hl1 = 10
n_nodes_hl2 = 10
n_classes = 2
batch_size = 256 
total_batches = int(1600000/batch_size)
num_epochs = 2

x = tf.placeholder('float')
y = tf.placeholder('float')

# define layers as dictionaries - wehere weights and biases are keys
hidden_1_layer = {'f_fum':n_nodes_hl1,
                'weights':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
                'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}


hidden_2_layer = {'f_fum':n_nodes_hl2,
                'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}


output_layer = {'f_fum':None,
                'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'biases':tf.Variable(tf.random_normal([n_classes]))}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
    return output

# create a saver object form tensorflow library
saver = tf.train.Saver()
tf_log = 'tf.log'


# need to save the model in form a checkpoints as time goes on using 
# tf.train.Saver()

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            # restore epoch from teh log file - if already logged
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1

        while epoch <= num_epochs:
            if epoch != 1:
                saver.restore(sess, 'model.ckpt')
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]
                    features = np.zeros(len(lexicon))
                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)

                    # if the batch size is reached
                    if len(batch_x) >= batch_size:
                        _, loss = sess.run([optimizer, cost], feed_dict={x:batch_x, y:batch_y})
                        epoch_loss += loss  # add batch loss to epoch loss
                        batch_x = []
                        batch_y = []
                        batches_run += 1
                        print('Batch run:',batches_run,'/',total_batches,'Epoch:',epoch,'| Batch Loss:',loss,)

            # after running the epoch save the model checkpoint
            saver.save(sess, 'model.ckpt')
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:', epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n')
            epoch += 1


def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            try:
                saver.restore(sess, 'model.ckpt')
            except Exception as e:
                print(str(e))
            epoch_loss = 0

        # get max value i.e. args max across axis 1 and compare
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        # convert list of True/False to float and average it
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))

                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
            print('Tested',counter,'samples.')
            test_x = np.array(feature_sets)
            test_y = np.array(labels)
            print('Accuracy:', accuracy_eval({x:test_x, y:test_y}))

def use_neural_network(input_data):
    # get output logits from the neural network model
    prediction = neural_network_model(x)
    print('Output Logits :',prediction)
   
    with open('lexicon.pickle', 'rb') as f:
        lexicon = pickle.load(f)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'model.ckpt')
        current_words = word_tokenize(input_data.lower())
        current_words = [lemmatizer.lemamtize(i) for i in current_words]
        features = np.zeros(len(lexicon))
        
        for word in current_words:
            if word.lower() in lexicon:
                index_value = lexicon.index(word.lower())
                features[index_value] += 1


        features = np.array(list(features))
        # pos : [1, 0], argmax: 0
        # neg : [0, 1], argmax: 1
        # run the prediction - with dictionary feed_dict - with input as features
        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[features]}), 1)))
        
        if result[0] == 0:
                print('Positive', input_data)
        elif result[0] == 1:
            print('Negative', input_data)


train_neural_network(x)
test_neural_network()
#use_neural_network('He\'s an idiot and jerk')
#use_neural_network('This was the best store i\'ve ever seen.')
