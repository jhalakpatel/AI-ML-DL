from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tempfile
import sys

def deepnn(x):
    """deepnn builds graph for deep net for classifiying digits

    Args:
        x: an input tensor with dimension (N_examples, 784), number of pixels in a standard MNIST image

    Returns:
        A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values equal to logits of classifying the digit into 10 classes. keep_prob is a scalar placeholder for probability of dropout or regularization
    """

    # reshape to use within a convolution neural net 
    # last dimension is for "features" - only one feature, since images are grayscale, it would be 3 if RGB image, 4 for RGBA
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])


    # first convolution layer - maps one grayscale image to 32 feature maps
    # due to parameter sharing for 5x5 depth slice with 32 depth, each neuron
    # in depth slice will share 5x5 weights thus total 5x5x1x32 weights
    # for each depth slice, there is one bias, thus for 32 depth slice, there
    # are 32 biases
    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # pooling layer - down samples by 2x
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    # second convolution layer - maps 32 features to 64 features
    # for 5x5x32 image --- due to parameter sharing, each depth slice of
    # 14x14x32x64 will share weights 5x5x32 for 64 depths
    # image = 14x14x32 : with kernel : 5x5 , 64 kernels
    # output 14x14x32x64 : with each depth slice 14x14x32 maps to 5x5x32
    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # second pooling layer
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)


    # fully connected layer, after convolution and downsampling - image 28x28 is down to 7x7x64 feature maps --> map this to 1024 features
    # with fully connected network - 7x7x64 output needs to be mapped to 1024 neurons
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        # reshape hpool2 layer to 7x7x64 dimension for fully connected layer
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        # convolution is replace with simple matrix multiplication to 
        # create fully connected layer of 1024 neuron output
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        

    # dropout layer - controls the complexity of model, prevent co-adaption
    # of features - we dont want trained model to overfit
    with tf.name_scope('dropout'):
        # to control the dropout rate
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


    # mapping 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024,10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # finally return y - i.e. hypothesis output
    return y_conv, keep_prob

def conv2d(x, W):
    """ return convolution layer with 2D strides """
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    """ max pool function will downsample a feature map by 2X """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def weight_variable(shape):
    """ generate weights of required shape """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ generate bias variable of given shape """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main(_):
    # import data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # create the model
    x = tf.placeholder(tf.float32, [None, 784]) # allow N number of samples

    # define loss and optimizer
    # correct inputs
    y_ = tf.placeholder(tf.float32, [None, 10])

    # build the graph and deep net
    y_conv, keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)


    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to : %s' % graph_location)

    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            # print the accuracy every 100 samples
            if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x:batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

if __name__ == '__main__':
    tf.app.run(main=main)






     



   

