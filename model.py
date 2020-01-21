import layers
import tensorflow as tf

filter_size = 31
strides = 2
input_shape = (16384, 1)
padding = 'valid'

filter_sizes = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

# Activation function
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)


def generative_net(input, filter_size, strides, padding, input_shape):
    net = layers._conv1d(inputs, 1, stride, paddig)
    net = layers._conv1d(inputs, 16, stride, paddig)
    net = layers._conv1d(inputs, 32, stride, paddig)
    net = layers._conv1d(inputs, 32, stride, paddig)
    net = layers._conv1d(inputs, 64, stride, paddig)
    net = layers._conv1d(inputs, 64, stride, paddig)
    net = layers._conv1d(inputs, 128, stride, paddig)
    net = layers._conv1d(inputs, 128, stride, paddig)
    net = layers._conv1d(inputs, 256, stride, paddig)
    net = layers._conv1d(inputs, 256, stride, paddig)
    net = layers._conv1d(inputs, 512, stride, paddig)
    net = layers._conv1d(inputs, 1024, stride, paddig)
    net = _leaky_relu(net)
    return net

def discreminator(input, filter_size, strides, padding, input_shape):
    net = layers._deconv1d(inputs, filters, stride, paddig)
    net = _leaky_relu(net)
    return net
