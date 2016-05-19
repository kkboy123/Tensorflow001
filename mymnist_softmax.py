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
from parser import read_data_sets
import tensorflow as tf


mnist = read_data_sets("data2", one_hot=True)
n_hidden_1 = 256
n_hidden_2 = 256
batch_size = 100
learning_rate = 1e-4

'''
Create the model
x, which is a 784-dim vector, is input layer
W, which is a 784x10 matrix, is the weight of the input layer
b, which is a 784-dim vector, is the bias of the input layer
y, which is a one-hot vector (10-dim), is the prediction
'''
def layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def output_layer(input, weight_shape, bias_shape):
    weight_init = tf.random_normal_initializer(stddev=(2.0/weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape,
                        initializer=weight_init)
    b = tf.get_variable("b", bias_shape,
                        initializer=bias_init)
    return tf.nn.softmax(tf.matmul(input, W) + b)

def inference(x):
    with tf.variable_scope("fc_1"):
        fc_1 = layer(x, [784, n_hidden_1], [n_hidden_1])

    with tf.variable_scope("fc_2"):
        fc_2 = layer(fc_1, [n_hidden_1, n_hidden_2], [n_hidden_2])

    with tf.variable_scope("output"):
        output = layer(fc_2, [n_hidden_2, 10], [10])

    return output

# def loss(output, y_):
#     loss = -tf.reduce_sum(y_ * tf.log(output))
#     return loss

def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    return loss

def training(cost, learning_rate, global_step):
    print("learning_rate is %r" % learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cost, global_step=global_step)
    return train_step

def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    output = inference(x)
    cost = loss(output, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = training(cost, learning_rate, global_step)
    eval_op = evaluate(output, y_)

    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    saver = tf.train.Saver()

    for i in range(300):
        print("epoch", i)
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            #accuracy = sess.run(eval_op, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            result = sess.run([eval_op, cost], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            accuracy = result[0]
            my_loss = result[1]

        print("Accuracy: ", accuracy)
        print("loss: ", my_loss)
    save_path = saver.save(sess, "/tmp/model2.ckpt")

    sess.close()
