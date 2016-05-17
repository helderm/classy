#!/env/bin python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main():

    # load minst data
    mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

    # placeholder value that we'll input when we ask TensorFlow to run a computation
    x = tf.placeholder(tf.float32, [None, 784])

    # initialize weights and biases as tf variables
    W = tf.Variable(tf.zeros([784, 10])) # 784 for input dimensions, 10 for classes
    b = tf.Variable(tf.zeros([10])) # one bias per class

    # model
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # placeholder for the correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])

    # cross entropy loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # gradient descent optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # initialize training
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    # train it!
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

if __name__ == '__main__':
    main()