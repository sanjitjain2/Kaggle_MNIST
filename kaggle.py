from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from numpy import array
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

LABELS = 10  # Number of labls(1-10)
IMAGE_WIDTH = 28  # Width/height if the image
COLOR_CHANNELS = 1  # Number of color channels

VALID_SIZE = 1000  # Size of the Validation data

EPOCHS = 20000  # Number of epochs to run
BATCH_SIZE = 32  # SGD Batch size
FILTER_SIZE = 5  # Filter size for kernel
DEPTH = 32  # Number of filters/templates
FC_NEURONS = 1024  # Number of neurons in the fully
# connected later
LR = 0.001  # Learning rate Alpha for SGD

FLAGS = None
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def deepnn(x):
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_WIDTH, COLOR_CHANNELS])

    # First convolution layer - maps one grayscale image to 8 feature maps.
    w1 = weight_variable([FILTER_SIZE, FILTER_SIZE, COLOR_CHANNELS, DEPTH])
    b1 = bias_variable([DEPTH])
    layer_conv1 = tf.nn.relu(conv_2d(x, w1) + b1)

    # Pooling layer - downsamples by 2X.
    layer_pool1 = max_pool_2x2(layer_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    w2 = weight_variable([FILTER_SIZE, FILTER_SIZE, DEPTH, DEPTH * 2])
    b2 = bias_variable([DEPTH * 2])
    layer_conv2 = tf.nn.relu(conv_2d(layer_pool1, w2) + b2)

    # Second pooling layer.
    layer_pool2 = max_pool_2x2(layer_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 100 features.
    wfc1 = weight_variable([IMAGE_WIDTH // 4 * IMAGE_WIDTH // 4 * 2 * DEPTH, FC_NEURONS])
    bfc1 = bias_variable([FC_NEURONS])

    flatten_pool2 = tf.reshape(layer_pool2, [-1, IMAGE_WIDTH // 4 * IMAGE_WIDTH // 4 * 2 * DEPTH])
    layer_fc1 = tf.nn.relu(tf.matmul(flatten_pool2, wfc1) + bfc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    # keep_prob = tf.placeholder(tf.float32)
    # layer_fc1_drop = tf.nn.dropout(layer_fc1,keep_prob)

    # Map the 100 features to 10 classes, one for each digit
    wfc2 = weight_variable([FC_NEURONS, LABELS])
    bfc2 = bias_variable([LABELS])

    y_conv = tf.matmul(layer_fc1, wfc2) + bfc2
    return y_conv
    # return y_conv,keep_prob


def conv_2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 down samples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    mnist = pd.read_csv('train.csv')
    labels = np.array(mnist.pop('label'))
    labels = LabelEncoder().fit_transform(labels)[:, None]
    labels = OneHotEncoder().fit_transform(labels).todense()

    mnist = StandardScaler().fit_transform(np.float32(mnist.values))
    mnist = mnist.reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH, COLOR_CHANNELS)
    train_data, valid_data = mnist[:-VALID_SIZE], mnist[-VALID_SIZE:]
    train_labels, valid_labels = labels[:-VALID_SIZE], labels[-VALID_SIZE:]

    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_WIDTH, COLOR_CHANNELS])

    y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv = deepnn(x)
    tf_pred = tf.nn.softmax(deepnn(x))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = 100 * tf.reduce_mean(correct_prediction)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ss = ShuffleSplit(n_splits=EPOCHS, train_size=BATCH_SIZE)
        ss.get_n_splits(train_data, train_labels)
        history = [(0, np.nan, 10)]  # Initial Error Measures
        for step, (idx, _) in enumerate(ss.split(train_data, train_labels), start=1):
            # fd = {x:train_data[idx], y_:train_labels[idx],  keep_prob: 0.5}
            fd = {x: train_data[idx], y_: train_labels[idx]}
            sess.run(train_step, feed_dict=fd)
            if step % 500 == 0:
                # fd = {x:valid_data, y_:valid_labels, keep_prob: 0.5}
                fd = {x: valid_data, y_: valid_labels}
                valid_loss, valid_accuracy = sess.run([cross_entropy, accuracy], feed_dict=fd)
                history.append((step, valid_loss, valid_accuracy))
                print('Step %i \t Valid. Acc. = %f \n' % (step, valid_accuracy))

        test = pd.read_csv('test.csv')
        test_data = StandardScaler().fit_transform(np.float32(test.values))  # Convert the dataframe to a numpy array
        test_data = test_data.reshape(-1, IMAGE_WIDTH, IMAGE_WIDTH,
                                      COLOR_CHANNELS)  # Reshape the data into 42000 2d images

        # fd = {x:test_data, keep_prob: 1.0}
        ss = ShuffleSplit(n_splits=28000, train_size=BATCH)
        ss.get_n_splits(test_data)
        test_labels = []
        for (idx, _) in enumerate(ss.split(test_data), start=0):
            temp_test_data = array(test_data[idx]).reshape(1, 28, 28, 1)
            fd = {x: temp_test_data}
            # image = array(img).reshape(1, 28,28,1)
            test_pred = sess.run(tf_pred, feed_dict=fd)
            temp = np.argmax(test_pred, axis=1)
            test_labels.append(temp)

        for i in range(len(test_labels)):
        	test_labels[i] = int(test_labels[i])

        submission = pd.DataFrame(data={'ImageId': (np.arange(test_labels.shape[0]) + 1), 'Label': test_labels})
        submission.index += 1
        submission.to_csv('submission.csv', index=False)
        #submission.tail()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
