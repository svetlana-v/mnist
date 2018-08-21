import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import trainer_tf_softmaxreg.model as model

sess = tf.InteractiveSession()


def run_experiment():
    mnist = input_data.read_data_sets("./data/", one_hot=True)

    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    y = model.build_model(x)

    sess.run(tf.global_variables_initializer())

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    for _ in range(1000):
        batch = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})


    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


 # Predictions for the test data
 #    test_prediction = tf.argmax(tf.nn.softmax(y),1)
 #    print(test_prediction.eval(feed_dict={x: mnist.test.images}))


if __name__ == '__main__':
    run_experiment()