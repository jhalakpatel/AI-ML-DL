import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_classes = 10
# batches of 100 images at a time and update the features
batch_size = 100

# height x width = 28x8 --> [None, 784] - reshape the input vector
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# stride = 1, 1, 1, 1 -- move conv by 1 dim
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# 2x2 pooling, moving 2 with stride
# when padding = 'SAME' - need to have exact those pixels to the edge - so that the
# the output dimension wont change due to convolution - padding = SAME will just 
# use the same boundary pixels
def maxpool2d(x):
                            # size of the window    # movement of the window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolution_neural_network(x):
    # 5x5 filter, take 1 input and produces 32 features
    weights = { 'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
                'W_conv2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
                'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
                'out':tf.Variable(tf.random_normal([1024,n_classes])) }

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
            'b_conv2':tf.Variable(tf.random_normal([64])),
            'b_fc':tf.Variable(tf.random_normal([1024])),
            'out':tf.Variable(tf.random_normal([n_classes])) }

    # reshape 784 to 28x28 image
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
   
    # hidden conv1 layer 1
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)
    output = tf.nn.relu(tf.matmul(fc, weights['out']) + biases['out'])
    return output

def train_neural_network(x,y):
    # prediction is one hot array
    prediction = convolution_neural_network(x)
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
            # iterate throough - 60000/100 - 600 batches of images
            # each batch of size = 100 images , total 600 batches - cycle
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # give the batch size for mnist data set
                # in other example - need to get our own batch generator
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # run the session to optimize the cost
                # tf takes care of modification of weights
                _, minibatch_loss = sess.run([optimizer, cost], feed_dict={ x:epoch_x, y:epoch_y })
                epoch_loss += minibatch_loss
            
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
        print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        # we are not telling it to feed the data through the model - need to figure out


train_neural_network(x,y)
