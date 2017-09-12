import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

# variables
num_epochs = 3
n_classes = 10
batch_size = 128
# for dnn - we pass all at once 
# for rnn - we pass in chunks - 28 chunks of 28
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    print('x original', x)
    x = tf.transpose(x, [1, 0, 2])
    print('x after transpose', x)
    x = tf.reshape(x, [-1, chunk_size])
    print('x after reshape', x)
    x = tf.split(x, n_chunks, 0)
    print('x after split',x)
    
    # create LSTM cell for the rnn_size of 128
    # will create 128 cells which are temporally connected?
    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # need to get the last layer outputs and multiply with weights and biases addn
    # matrix multiplication of the final output and weights + biases
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output

def train_rnn(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

    
        # iterate through all the epochs
        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)

                # convert image into chunk of 28 chunks of size  28 for each batch image
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                
                _, batch_loss = sess.run([optimizer, cost], feed_dict = {x:epoch_x, y:epoch_y})
                epoch_loss += batch_loss

            print('Epoch', epoch, 'completed out of', num_epochs,'loss:', epoch_loss)

        # prediction will be called with x input test images
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:', accuracy.eval({x:mnist.test.images.reshape(-1, n_chunks, chunk_size), y:mnist.test.labels}))

train_rnn(x)
