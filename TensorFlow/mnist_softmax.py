from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main(_):
    # import mnist data set
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # create model - softmax regression - classifier
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x,W) + b              # will get estimated ys

    # define loss - cross entropy loss function and optimizer - schocastic gradient descent - for backpropogation
    y_ = tf.placeholder(tf.float32, [None, 10])     # true values or correct values

    # raw formulation of cross entropy
    # tf.reduce_mean(tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), reduction_indices=[1]))
    # above function can be numerically unstable

    # get loss function on using sofmax cross entropy on raw y and labels
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # get the training setup to minimize the loss funciton using gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # perform training in the batch of 100 - Gradient descent is called stchocastic gradient descent when done in bacthes - 
    # random 100 samples or data points are used to perform gradient descent - or train with backprop to udate W and b
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})           # train with xs and correct labels, evidence y is nothing but manifestation of x


    # test the trained model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))      # argmax will find the max of a tensor along a given axis
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))


if __name__ == '__main__':
    tf.app.run(main=main)
