"""Functions for reading Runway data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', '../data/release_runway_0.2/data/images/',
                           """Path to the Runway data directory.""")
tf.app.flags.DEFINE_string('data_dir', '../data/release_runway_0.2/data/',
                           """Path to the serialized Runway dataset.""")
tf.app.flags.DEFINE_string('test_share', 0.2,
                           """Share of the dataset that will be saved for testing.""")
tf.app.flags.DEFINE_string('validation_share', 0.05,
                           """Share of the dataset that will be saved for validation.""")
tf.app.flags.DEFINE_string('max_images', 4096,
                           """Max number of images imported. 0 for no limit.""")

# constants of the runway dataset
IMAGE_WIDTH = 142
IMAGE_HEIGHT = 302

# other values
_FILENAME_TRAIN = 'train.tfrecords'
_FILENAME_TEST = 'test.tfrecords'
_FILENAME_VAL = 'val.tfrecords'

def _convert_to_record(image, label, writer):
    """
    Serialize a single image and label and write it
    to a TFRecord
    :param image: 3D image tensor
    :param label: 1D label tensor
    :param writer: file writer
    """
    image_raw = image.tostring()
    rows = image.shape[1]
    cols = image.shape[2]
    depth = image.shape[3]
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
        'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
        }))
    writer.write(example.SerializeToString())


def serialize():
    """
    Serialize the Runway images and labels into TFRecords
    """
    base_data_path = FLAGS.input_dir
    images_paths = []

    # get all images paths
    for brand in os.listdir(base_data_path):
        brand_path = os.path.join(base_data_path, brand)
        for fashionshow in os.listdir(brand_path):
            fashionshow_path = os.path.join(brand_path, fashionshow)
            image_files = glob.glob(fashionshow_path + '/*.jpg')
            images_paths.extend(image_files)

        if FLAGS.max_images and len(images_paths) > FLAGS.max_images:
            images_paths[:FLAGS.max_images]
            break

    # shuffle the images
    np.random.shuffle(images_paths)

    # set the queue
    filename_queue = tf.train.string_input_producer(images_paths)

    # calculate how many images each set will have
    num_test_images = int(len(images_paths) * FLAGS.test_share)
    num_val_images = int(len(images_paths) * FLAGS.validation_share)
    num_train_images = len(images_paths) - num_test_images - num_val_images

    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)

    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # write training images
        filename = os.path.join(FLAGS.data_dir, _FILENAME_TRAIN)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(num_train_images):
            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, np.array([0]), writer)
        writer.close()

        # write test images
        filename = os.path.join(FLAGS.data_dir, _FILENAME_TEST)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(num_test_images):
            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, np.array([0]), writer)
        writer.close()

        # write val images
        filename = os.path.join(FLAGS.data_dir, _FILENAME_VAL)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(num_val_images):
            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, np.array([0]), writer)
        writer.close()

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)

def _read_and_decode(filename_queue):
    """
    Reads and deserialize a single example from the queue
    :param filename_queue: files to read
    :return: image, label
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * 3])
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def inputs(batch_size):
    """
    Read images and labels in shuffled batches
    :param batch_size: size of the batch
    :return: images 4D tensor, labels 1D tensor
    """
    filename = os.path.join(FLAGS.data_dir, _FILENAME_TRAIN)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename])

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = _read_and_decode(filename_queue)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, sparse_labels


if __name__ == '__main__':
    serialize()
