import tensorflow as tf
from tensorflow import keras

def _dense(inputs):
    return tf.layers.dense(inputs)

def _batch_norm(inputs, is_training):
    return tf.layers.batch_normalization(inputs, momentum=0.999, epsilon=0.001, training=is_training)

def _deconv1d(inputs, filters, stride, padding='same'):
    return tf.nn.conv1d_transpose(inputs, filters, stride, padding)

def _conv1d(inputs, filters, stride, paddig='same'):
    return tf.nn.conv1d(inputs, filters, stride, padding)