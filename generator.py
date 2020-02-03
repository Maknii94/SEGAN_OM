from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def downsample(x, filter_width, kernel=31,
               strides=2, padding='same', init=None):
    """
    """

    batch, in_width, in_channels = x.shape
    filters = = tf.Variable(tf.random_normal([kernel, in_channels, filter_width],
                                             stddev=0.35), name="filters")

    # convolution layer
    block = tf.nn.conv1d(inptut=x, filter=filters, stride = strides,
                         padding = padding)
    return block
