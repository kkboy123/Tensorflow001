# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
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


mnist = input_data.read_data_sets("data", one_hot=True)
sess = tf.InteractiveSession()

'''
Create the model
x, which is a 784-dim vector, is input layer
W, which is a 784x10 matrix, is the weight of the input layer
b, which is a 784-dim vector, is the bias of the input layer
y, which is a one-hot vector (10-dim), is the prediction
'''
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.random_normal([784, 128], stddev=0.3))
b = tf.Variable(tf.random_normal([128], stddev=0.3))
y = tf.nn.softmax(tf.matmul(x, W) + b)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(y, keep_prob)

W2 = tf.Variable(tf.random_normal([128, 10], stddev=0.1))
b2 = tf.Variable(tf.random_normal([10], stddev=0.1))
y2 = tf.nn.softmax(tf.matmul(h_fc1_drop, W2) + b2)

'''
Define loss and optimizer
'''
y_ = tf.placeholder(tf.float32, [None, 10])
'''
cross_entropy
'''
loss = -tf.reduce_sum(y_ * tf.log(y2))
'''root mean square'''
# loss = tf.reduce_mean(tf.square(y2 - y_))
'''
different training method
https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#optimizers
'''
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.01,0.01).minimize(loss)

tf.initialize_all_variables().run()

'''
Training 1000 times
Mini batch
batch size = 100
'''
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.5})

''' Test trained model'''
correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
sess.close()













'''
Exercise 1
try different loss function and optimizer

Exercise 2
add another layer
'''







