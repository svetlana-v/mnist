import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import trainer_tf_with_LRD.model as model

sess = tf.InteractiveSession()


def run_experiment(args):
    """Run the training and evaluate the model"""

    # Import data
    mnist = input_data.read_data_sets("./data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784], name='Input_data')
    y_ = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
    keep_prob = tf.placeholder(tf.float32, name='Keep_prob')

    tf_alpha = tf.placeholder(tf.float32, shape=None)
    decay_rate = 0.001
    alpha0 = 0.0005


    # Build the model
    y_conv = model.build_model(x, keep_prob)

    # For predicting labels
    label = tf.argmax(tf.nn.softmax(logits=y_conv), 1)

    # Add cross entropy to Tensorboard by tf.summary.scalar
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)


    # Train the Model
    train_step = tf.train.AdamOptimizer(tf_alpha).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)


    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.job_dir + '/output/train', sess.graph)
    test_writer = tf.summary.FileWriter(args.job_dir + '/output/test')

    tf.global_variables_initializer().run()


    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        alpha = 1 / (1 + decay_rate * i) * alpha0
        if i % 100 == 0:
            summary, train_accuracy = sess.run([merged, accuracy], feed_dict={
                    x:batch[0], y_: batch[1], keep_prob: 1.0, tf_alpha: alpha})
            test_writer.add_summary(summary, i)
            print("step %d, training accuracy %g" % (i, train_accuracy))
        # write summaries
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, tf_alpha: alpha})
        train_writer.add_summary(summary, i)

    # Evaluate the Model
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Save model
    builder = tf.saved_model.builder.SavedModelBuilder(args.job_dir + '/export')

    # Build the signature_def_map
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_keep_prob = tf.saved_model.utils.build_tensor_info(keep_prob)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(label)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'images': tensor_info_x, 'keep_prob': tensor_info_keep_prob},
            outputs={'labels': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_images':
                prediction_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    builder.save()

    print('Done exporting!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    args = parser.parse_args()

    run_experiment(args)

