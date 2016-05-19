# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#                 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape, name='None'):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name='None'):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial,)


def conv2d(_x, _w):
    return tf.nn.conv2d(_x, _w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(_x):
    return tf.nn.max_pool(_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("data", one_hot=True)
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# reshape x for new W
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional layer l
with tf.name_scope("conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    b_conv1 = bias_variable([32], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
with tf.name_scope("conv2") as scope:
    W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    b_conv2 = bias_variable([64], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

# Fully (densely) connected Layer
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
    b_fc1 = bias_variable([1024], name='b_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout for reducing over-fitting
with tf.name_scope("dropout") as scope:
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer
with tf.name_scope("sftmax") as scope:
    W_fc2 = weight_variable([1024, 10], name='W_fc2')
    b_fc2 = bias_variable([10], name='b_fc2')
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# Define loss and optimizer
with tf.name_scope("xent") as scope:
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
with tf.name_scope("train") as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Train and test the model
with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy_summary = tf.scalar_summary("accuracy", accuracy)

# Add summary ops to collect data
W_conv1_hist = tf.histogram_summary("W_conv1_summary", W_conv1)
b_conv1_hist = tf.histogram_summary("b_conv1_summary", b_conv1)
h_conv1_img = list()
for _i, h1_i in enumerate(tf.split(3, 32, h_conv1)):
    h_conv1_img.append(tf.image_summary("h_conv1_%s" % _i, h1_i))
W_conv1_hist = tf.histogram_summary("W_conv2_summary", W_conv2)
b_conv1_hist = tf.histogram_summary("b_conv2_summary", b_conv2)
h_conv2_img = list()
for _i, h2_i in enumerate(tf.split(3, 64, h_conv2)):
    h_conv2_img.append(tf.image_summary("h_conv2_%s" % _i, h2_i))
W_fc1_hist = tf.histogram_summary("W_fc1_summary", W_fc1)
b_fc1_hist = tf.histogram_summary("b_fc1_summary", b_fc1)
W_fc2_hist = tf.histogram_summary("W_fc2_summary", W_fc2)
b_fc2_hist = tf.histogram_summary("b_fc2_summary", b_fc2)


# Merge all the summaries and write them out to /tmp/mnist_conv_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_conv_logs", sess.graph_def)
init = tf.initialize_all_variables()

# Launch the graph
sess.run(init)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Restore variables from disk
saver.restore(sess, "/Users/will/HelloPycharm/tutorial/export/mnist_conv_logs/model.ckpt")
# Print the final accuracy
print ("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()
