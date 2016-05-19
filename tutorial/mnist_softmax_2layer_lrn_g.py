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
import numpy as np


def weight_variable(shape, name='None'):
    initial = tf.truncated_normal(shape, stddev=0.05, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name='None'):
    initial = tf.truncated_normal(mean=0.092, stddev=0.002, shape=shape, name=name)
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
c1_input = 64
c2_input = 128
fc_input = 1024

# reshape x for new W
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional layer l
with tf.name_scope("conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, c1_input], name='W_conv1')
    b_conv1 = bias_variable([c1_input], name='b_conv1')
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

# second convolutional layer
with tf.name_scope("conv2") as scope:
    W_conv2 = weight_variable([5, 5, c1_input, c2_input], name='W_conv2')
    b_conv2 = bias_variable([c2_input], name='b_conv2')
    h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_norm2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

# Fully (densely) connected Layer
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([7 * 7 * c2_input, fc_input], name='W_fc1')
    b_fc1 = bias_variable([fc_input], name='b_fc1')
    h_pool2_flat = tf.reshape(h_norm2, [-1, 7*7*c2_input])
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
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # logloss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y_conv, 1e-25, 1.0)))
    logloss = -tf.reduce_sum(y_*tf.log(y_conv))
    logloss2 = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)), 1))
    # ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
with tf.name_scope("train") as scope:
    train_step = tf.train.AdamOptimizer(1e-4).minimize(logloss)
    # train_step = tf.train.AdagradOptimizer(1e-4).minimize(logloss)

# Train and test the model
with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # accuracy_summary = tf.scalar_summary("accuracy", accuracy)

init = tf.initialize_all_variables()

# Launch the graph
sess.run(init)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

for i in range(6001):
    # batch = mnist.train.next_batch(50)
    batch = mnist.test.next_batch(50)
    if i % 100 == 0:
        feed = {x: batch[0], y_: batch[1], keep_prob: 1.0}
        # train_accuracy = accuracy.eval(feed_dict=feed)
        # print ("step %d, training accuracy %g" % (i, train_accuracy))
        # result = sess.run([merged, accuracy, cross_entropy], feed_dict=feed)
        result = sess.run([accuracy, logloss, logloss2, y_conv], feed_dict=feed)
        # summary_str = result[0]
        acc = result[0]
        loss = result[1]
        # writer.add_summary(summary_str, i)
        print("Accuracy at step %s: %s; loss : %s" % (i, acc, loss))
        if np.isnan(loss):
            print("logloss2 : %s" % result[2])
            print("y_conv : %s" % result[3])
            result = sess.run([W_conv1, W_conv2, W_fc1], feed_dict=feed)
            print("W_conv1 : %s" % result[1])
            print("W_conv2 : %s" % result[2])
            print("W_fc1 : %s" % result[3])
            break
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Print the final accuracy
# print ("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
total_logloss = 0.0
for i in range(5):
    start = i * 10000
    end = (i + 1) * 10000
    feed = {x: mnist.train.images[start:end], y_: mnist.train.labels[start:end], keep_prob: 1.0}
    print ("test accuracy %g" % accuracy.eval(feed_dict=feed))
    result = sess.run(logloss2, feed_dict=feed)
    print ("logloss : %s" % result)
    total_logloss += result
# Save the variables to disk.
save_path = saver.save(sess, "/tmp/mnist_conv_lrn_logs/model.ckpt")
print("Model saved in file: %s" % save_path)
total_logloss /= 5.0
print ("total logloss : %s" % total_logloss)
sess.close()
