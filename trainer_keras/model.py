import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data


def build_model(units_l1=128, units_l2=10):
    # Build a model
    model = keras.Sequential([
        keras.layers.Dense(units_l1, activation=tf.nn.relu),
        keras.layers.Dense(units_l2, activation=tf.nn.softmax)
    ])
    return model
