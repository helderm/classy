"""Functions for reading Runway data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle

import os
import glob
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('images_dir', '../data/release_runway_0.2/data/images/',
                           """Path to the Runway data directory.""")
tf.app.flags.DEFINE_string('output_dir', '../data/release_runway_0.2/data/',
                           """Path to the serialized Runway dataset.""")
tf.app.flags.DEFINE_string('test_share', 0.2,
                           """Share of the dataset that will be saved for testing.""")
tf.app.flags.DEFINE_string('validation_share', 0.05,
                           """Share of the dataset that will be saved for validation.""")
tf.app.flags.DEFINE_string('max_images', 131072,
                           """Max number of images imported. 0 for no limit.""")
tf.app.flags.DEFINE_integer('train_num_files', 24,
                            """ Number of training files to have.""")


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

    if image.shape[1] != IMAGE_HEIGHT or image.shape[2] != IMAGE_WIDTH or image.shape[3] != 3:
        print('Found an image with wrong dimensions, ignoring...')
        return

    image_raw = image.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
        }))
    writer.write(example.SerializeToString())


def serialize():
    """
    Serialize the Runway images and labels into TFRecords
    """
    base_data_path = FLAGS.images_dir
    images_paths = []
    images_labels = []

    # get all images paths
    for brand in os.listdir(base_data_path):
        brand_path = os.path.join(base_data_path, brand)
        for fashionshow in os.listdir(brand_path):
            #get labels
                #0: Spring
                #1: Fall
                #2: Resort
                #3: Pre-fall
            image_label = -1
            if (fashionshow[0] == 'S'):
                image_label = 0
            elif (fashionshow[0] == 'F'):
                image_label = 1
            elif (fashionshow[0] == '2'):
                if (fashionshow[4] == 'R'):
                    image_label = 2
                if (fashionshow[4] == 'P'):
                    image_label = 3
            if (image_label == -1):
                raise ValueError('Label could not be identified')

            fashionshow_path = os.path.join(brand_path, fashionshow)
            image_files = glob.glob(fashionshow_path + '/*.jpg')
            images_paths.extend(image_files)
            for i in range (len(image_files)):
                images_labels.append(image_label)

        if FLAGS.max_images and len(images_paths) > FLAGS.max_images:
            images_paths[:FLAGS.max_images]
            images_labels[:FLAGS.max_images]
            break

    if (len(images_paths) != len(images_labels)):
        raise ValueError('Inputs and outputs have different lengths')

    # shuffle the images
    images_paths_shuf = []
    images_labels_shuf = []
    index_shuf = list(range(len(images_paths)))
    shuffle(index_shuf)
    for i in index_shuf:
        images_paths_shuf.append(images_paths[i])
        images_labels_shuf.append(images_labels[i])

    # set the queue
    filename_queue = tf.train.string_input_producer(images_paths_shuf, shuffle=False)

    # calculate how many images each set will have
    num_test_images = int(len(images_paths_shuf) * FLAGS.test_share)
    num_val_images = int(len(images_paths_shuf) * FLAGS.validation_share)
    num_train_images = len(images_paths_shuf) - num_test_images - num_val_images

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

        # split images in different files
        num_examples_per_file = int(num_train_images / FLAGS.train_num_files)
        file_idx = 0

        # write training images
        filename = os.path.join(FLAGS.output_dir, _FILENAME_TRAIN + str(file_idx))
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(num_train_images):
            if i % 1000 == 0:
                print('- Wrote {0} out of {1} images...'.format(i, num_train_images))

            if (i+1) % num_examples_per_file == 0 and file_idx+1 != FLAGS.train_num_files:
                writer.close()
                file_idx += 1
                filename = os.path.join(FLAGS.output_dir, _FILENAME_TRAIN + str(file_idx))
                print('Writing', filename)
                writer = tf.python_io.TFRecordWriter(filename)

            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, images_labels_shuf[i], writer)
        writer.close()

        # write test images
        filename = os.path.join(FLAGS.output_dir, _FILENAME_TEST)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(num_train_images,(num_train_images + num_test_images)):
            if (i - num_train_images) % 1000 == 0:
                print('- Wrote {0} out of {1} files...'.format(i - num_train_images, num_test_images))
            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, images_labels_shuf[i], writer)
        writer.close()

        # write val images
        filename = os.path.join(FLAGS.output_dir, _FILENAME_VAL)
        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range((num_train_images + num_test_images),(num_train_images + num_test_images + num_val_images)):
            if (i - (num_train_images + num_test_images)) % 1000 == 0:
                print('- Wrote {0} out of {1} files...'.format(i - (num_train_images + num_test_images),
                                                               num_val_images))
            image_tensor = np.array(sess.run([image]))
            _convert_to_record(image_tensor, images_labels_shuf[i], writer)
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

    # Convert from a scalar string tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_WIDTH * IMAGE_HEIGHT * 3])
    image = tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, 3])

    # resizing the image to 50% of the original width
    image = tf.image.resize_images(image, 151, 71, method=0, align_corners=False)

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(batch_size, num_examples_epoch, eval_type='train', shuffle=True):
    """
    Read images and labels in  batches
    :param batch_size: size of the batch
    :return: images 4D tensor, labels 1D tensor
    """
    if eval_type == 'train':
        filenames = [os.path.join(FLAGS.output_dir, _FILENAME_TRAIN + str(i))
                        for i in range(FLAGS.train_num_files)]
                        #for i in range(1)]
    elif eval_type == 'test':
        filenames = [os.path.join(FLAGS.output_dir, _FILENAME_TEST)]
    else:
        filenames = [os.path.join(FLAGS.output_dir, _FILENAME_VAL)]


    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = _read_and_decode(filename_queue)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_epoch *
                                 min_fraction_of_examples_in_queue)

        if shuffle:
            # Shuffle the examples and collect them into batch_size batches.
            # (Internally uses a RandomShuffleQueue.)
            # We run this in two threads to avoid being a bottleneck.
            images_batch, labels_batch = tf.train.shuffle_batch(
                [image, label], batch_size=batch_size, num_threads=2,
                capacity=min_queue_examples + 3 * batch_size,
                # Ensures a minimum amount of shuffling of examples.
                min_after_dequeue=min_queue_examples    )
        else:
            images_batch, labels_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=2,
                capacity=min_queue_examples + 3 * batch_size)

    return images_batch, labels_batch


if __name__ == '__main__':
    serialize()
