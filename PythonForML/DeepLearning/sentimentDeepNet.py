import tensorflow as tf
import numpy as np
# import custom create feature set function
from createSentimentFeatureSet import create_feature_sets_and_labels

train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 2

# batches of 100 images at a time and update the features
batch_size = 100

# all the training data of is of same length - i.e. feature length
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    # use random number - will work with some standard deviation and mena
    # biases are added in after the multiplication - simply add to the output layer
    # (input_data * weights) + bias
    # bias is useful if all the input data is zero - no neuron will fire without bias
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases':tf.Variable(tf.random_normal([n_classes]))}

    # define the model
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # threshold - activation function
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']),  hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    return output


def train_neural_network(x,y):
    # prediction is one hot array
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    # minimize the cost
    # learning rate - for the optimizer
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # epochs - cycles of the forward and backpropogation
    num_epochs = 10

    # start the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # run training for number of epochs
        for epoch in range(num_epochs):
            epoch_loss = 0
            # _ is for variable which we dont care about
            # for each epoch - forward and backprop
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                _, minibatch_loss = sess.run([optimizer, cost], feed_dict={ x:batch_x, y:batch_y })
                epoch_loss += minibatch_loss
                i += batch_size
            # print per epoch stats
            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:',epoch_loss)

        # Done with training, above
        # tf.argmax - return maximum value index - along axis = 1, 
        # i.e. across all the cols
        # correct is a simple list of boolean - 
        # i.e. True or False for the prediction
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        
        # evaluation
        # convert boolean to float
        # find out the average accuracy - using total true
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x,y:test_y}))
        # we are not telling it to feed the data through the model - need to figure out

train_neural_network(x,y)

"""
1. Dataset(60 K Images) - MNIST - 28x28 images, classify them from 0-9D
2. each feature - is pixel value i.e. 0 or 1

3. Network graph logic
    1. input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (actiation function) > weights > output layer

    2. Simple feed forward network logic - compare the output with intended output --> cost function - cross entropy 

    3. optimization function (optimizer) --> minimize cost (AdamOptimizer.. SGD, AdaGrad)

    4. backpropogation 

4. feed forward + backprop == epoch - for a given input sample - perform feed forward propogation and backpropogation - complete cycle is called epoch
"""
