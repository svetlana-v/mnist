import multiprocessing
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


"""Parameters Initialization"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='Weights')


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='Bias')


"""Convolution and Pooling definition"""

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def build_model(x, keep_prob):
    """Building a Multilayer Convolutional Network"""

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First Convolutional Layer
    with tf.name_scope('1_32C5_MP2'):
        with tf.name_scope('weights'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            variable_summaries(W_conv1)
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)


    # Second Convolutional Layer
    with tf.name_scope('2_64C5_MP2'):
        with tf.name_scope('weights'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            variable_summaries(W_conv2)
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)


    # Densely Connected Layer
    with tf.name_scope('3_1024N'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024])
            variable_summaries(W_fc1)
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    # Dropout (To reduce overfitting)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Readout Layer
    with tf.name_scope('4_10N'):
        with tf.name_scope('weights'):
            W_fc2 = weight_variable([1024, 10])
            variable_summaries(W_fc2)
        b_fc2 = bias_variable([10])

        # Prediction
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_conv




