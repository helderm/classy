"""Functions for reading Runway data."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import signal
import re
import numpy
import tensorflow as tf
import time
import numpy as np
from datetime import datetime

import runway as rw
import classy as cl

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '../data/train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_boolean('compress_output', False,
                            """Whether to log device placement.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
TOWER_NAME = 'tower'
NUM_CLASSES = 4

# Constants describing the training process.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# set to true when the user requested to stop the training
_shutdown = False

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def train(total_loss, global_step):
    """Train the Classy model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.scalar_summary('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = cl._add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
      cl.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def run_training():
    """
    Train the Classy model for a number of steps
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for runway
        images, labels = rw.inputs(FLAGS.batch_size, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cl.inference(images)

        # Calculate loss.
        loss = cl.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()

        # get a tensor for evaluating the training
        correct_examples = evaluation(logits, labels)

        # Build an initialization operation to run below.
        init = tf.initialize_all_variables()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        max_steps = np.iinfo(np.int32).max if FLAGS.max_steps == 0 else FLAGS.max_steps
        for step in range(max_steps):
            start_time = time.time()
            _, loss_value, num_corr_examples = sess.run([train_op, loss, correct_examples])
            duration = time.time() - start_time

            #images_v, labels_v = sess.run([images, labels])
            #print(images_v)
            #print('--------')
            #print(labels_v)

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f, train_acc = %d/%d, (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, num_corr_examples, FLAGS.batch_size,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if _shutdown:
                break

        print('Classy training finished!')

        if FLAGS.compress_output:
            import shutil
            outname = 'train_cl_{0}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
            shutil.make_archive(outname, 'gztar', FLAGS.train_dir)


def handler(signum, frame):
    global _shutdown
    print('Classy training shutdown requested! Finalizing...')
    _shutdown = True


def main(argv=None):
    # register signal handlers
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    run_training()

if __name__ == '__main__':
    tf.app.run()
