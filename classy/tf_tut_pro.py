#!/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():
    """ Softmax Regression model """

    # load the mnist data
    mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

    # start a session
    sess = tf.InteractiveSession()

    # setup input / true output nodes
    #   x: batch_size * flattened_image tensor
    x = tf.placeholder(tf.float32, shape=[None, 784])
    #   y_: batch_size * #classes (one-hot)
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    # define weigths and biases
    #   W: 784 input_features * 10 outputs tensor
    W = tf.Variable(tf.zeros([784,10]))
    #   b: 10 outputs tensor
    b = tf.Variable(tf.zeros([10]))

    # initialize variables for session
    sess.run(tf.initialize_all_variables())

    # regression model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # loss function to be minimized
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # setup the gradient descent optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # train the model
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # evaluate our model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print test accuracy
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # around 92%, that is pretty bad for mnist
    # lets improve that with a ConvNet!

    # first, define some initialization functions
    def weight_variable(shape):
        """ Initialize weights with a bit of noise """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """ Initialize biases with a bit of noise """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # define the convolution and pooling operators
    def conv2d(x, W):
        """ Convolution, stride 1, zero padded """
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        """ Max pooling 2*2 """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # define weights and biases for the first convolutional layer
    #   W_conv1: 5*5 kernel, 1 feature channel, 32 kernels
    W_conv1 = weight_variable([5, 5, 1, 32])
    #   b_conv1: 32 biases, one for each kernel
    b_conv1 = bias_variable([32])

    # reshape the image back to 2D (actually a 4D K*H*W*C tensor)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # apply the convolution, followed by ReLU and max pooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second layer!
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # third is a fully connected layer
    #   image has size 7*7 now
    #   1024 neurons to process the whole image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    # flat out the image to a vector
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # multiply everything by weight plus bias, apply the ReLU
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # setup droput before the readout layer, to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # add the final readout layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # train and evaluate model
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.initialize_all_variables())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    main()