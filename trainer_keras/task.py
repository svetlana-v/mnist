import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data

import trainer_keras.model as model

""" Keras is a Higher-level APIs based on Tensorflow. 
It hides the details of graphs and sessions from the end user. """


def run_experiment():
    # Load data
    mnist = input_data.read_data_sets("./data/", one_hot=False)

    # Build the model
    classificator = model.build_model()


    exporter = tf.estimator.FinalExporter('mnist',
                                          example_serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=hparams.eval_steps,
                                      exporters=[exporter],
                                      name='census-eval'
                                      )

    # Set required settings
    classificator.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    classificator.fit(mnist.train.images, mnist.train.labels, epochs=5)

    # Evaluate accuracy
    test_loss, test_acc = classificator.evaluate(mnist.test.images, mnist.test.labels)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    # Run the training job
    run_experiment()
