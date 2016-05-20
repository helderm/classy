#!/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import runway
import net

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('train', True,
                            """Whether to train the model or to parse the data.""")

def main():
    if FLAGS.train:
        net.run_training()
    else:
        runway.serialize()

if __name__ == '__main__':
    main()