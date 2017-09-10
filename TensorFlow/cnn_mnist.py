from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """ Model function for CNN"""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1,28,28,1])
    """ 
        input layer : monochrome 28x28 pixel image
        convert input image to : [batch_size, 28, 28, 1]
        batch size = -1 : dynamically computed basised on input values in feature "x"
        batch_size is treated as hyper parameter - can be tuned
        eg. for batch of 100, [100, 28, 28, 1]
    """
    # Convolution Layer #1
    """     
        Woutput = Winput - FilterDim + 2*Padding / Strides + 1
        with padding = same, will add "0" values to the boundary
        to preserve tensor widht and height of tensor to 28
        otherwise, 5x5 filter over 28x28 image will produce : 24x24 images
        
        simple tf.nn.relu activation function applied to the feature maps

        output tensor shape : [batch, 28, 28, 32] : now 32 channels holding the output from each filters - due to parameter sharing - weights = 5x5x32 - i.e. each depth slice will share the 5x5 filter weights for 32 depths - huge computation saving
        otherwise, for each output neuron - 28x28x32 x 5x5 will be number fo weights
        plus bias will be simple 1D tensor of size 32

    """
    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    # Pooling Layer #1
    """
        maxpooling with stride = 2, for different stride for width and heigh = [3,6]
        new shape : [batch_size, 14, 14, 32]
    """
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    # Convolution Layer #2 and Pooling Layer #2
    """
        with 64, 5x5 filters with ReLU activation  and Pooling layer : 
        conv2 shape : [batch_size, 14, 14, 32] : convolve 14x14x32 layer with 5x5 - each filter will produce 14x14 size of depth slice, simple slice and multiply add - 
        assumtion for convolve each depth of 32, will be added for single sliding of 5x5 over 14x14 depth slice
        with weights = 5x5x64, bias = 1d tensor for size 64
        with max pooling : 7x7x64 : conv2 shape
    """
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5,5],
            padding="same",
            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    # Dense Layer
    """
        with 1024 layers and ReLU activation - CNN peform classification on feature extracted by convolution and pooling layers
        Need to flaten the feature map to shape : [batch_size, features]
        i.e. convert tensor dimension from 4 to 2
        [-1, 7x7x64] - simple reshape
        features dimension to have value of 7x7x64 : 3136 : [batchsize, 3136]
        now connect dense layers - i.e. 3136 with 1024 layer neurons

        add dropout regularization to dense layer : dropout rate : 0.4, 40% of elements will be randomly droppped out during training
        dropout only in training mode, not in evaluation or prediction as we want to prevent overfitting during the training stage
        output of dropout : [batchsize, 1024]

    """

    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    dense = tf.layers.dense(
                inputs=pool2_flat, 
                units=1024, 
                activation=tf.nn.relu)

    dropout = tf.layers.dropout(
                inputs=dense,
                rate=0.4,
                training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer - final layer where number of neurons are equal to output classes
    """
        return raw values of our predictions - dense layer with 10 neurons
        output : [batch_size, 10]
    """
    
    logits = tf.layers.dense(inputs=dropout, units=10)


    """
        convert predictions to predicted class : 
            row of logits tensor with highest or greatest value - use argmax to get the maximum value in a tensor

        
        convert predictions to probabilities :
        to get the probs, simply app softmax activation on the logits tensor to get the output tensor

        store the results into a dictionary - predictions : with keys as classes and probabilities

    """
    predictions = {
            # Generate Predictions for PREDICT and EVAL mode)
            # get the maximum across axis 1, i.e. max of probs for a prediction
            "classes" : tf.argmax(input=logits, axis=1),
            "probabilities" : tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode== tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)



    """ 
        calculating losses : for training and evaluation we need loss function
        for prediciton loss function is not required

        cross entropy is standard function for multi class classifiction
        to calculate loss, first convert the labels into one hot encoding

        now compute cross entropy of onehot_labels and softmax predictions from our logits layer : calculate cross entropy and return loss as scalar tensor

    """


    # Calculate Loss function for TRAIN and EVAL modes
    # convert lables to one hot vectors of depth = 10
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    
    # simple cross entropy function to calculate the loss
    # cross entropy - measure of how incorrect is the prediction - 
    # idea from information theory
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)


    """ 
        configuring training:
        eval function will simply calculate the loss based on current model parameters, weights and biases, but during training we want to minimize loss and optimize those parameters

    with learning rate alpha = 0.001 use gradientdescent optimizer to minimize loss

    """

    # Configure the training Op for TRAIN mode
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)    
    """
        for eval mode : we need to get the predicted class and get the accuracy

    """

    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


""" Load Training and Test Data """
"""
    
"""
def main(unused_argv):
    # load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images     # returns a np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images       # returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # estimator needs model function and directory to store the mode
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, 
            model_dir="/Users/jhalakpatel/Desktop/Machine_Learning/TensorFlow/source/mnist_convnet_model")
    
    # logging to track progress in CNN 
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
            
    
    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x" : train_data},
        y=train_labels,
        batch_size=100,     # minibatch size of 100, training samples
        num_epochs=None,    # model will train unitla specified steps are reached
        shuffle=True)       # will shuffle training data

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=1000,
        hooks=[logging_hook])


    # evaluate the model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x":eval_data},
        y=eval_labels,
        num_epochs=1,       # model evaluates metrics over one epoch of data and return result, epochs : single pass through entire training dataset
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
            

if __name__ == "__main__":
    tf.app.run()
