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

from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from parser import read_data_sets
import numpy as np

# mnist = input_data.read_data_sets("data", one_hot=True)
mnist = read_data_sets("data2", one_hot=True)
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
    weight_init = tf.random_normal_initializer(stddev=(2.0 / weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)


def conv2d(input, weight_shape, bias_shape):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming)**0.5)
    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(input, k=2):
    return tf.nn.max_pool(
        input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def inference(x, keep_prob):

    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(x, [5, 5, 1, 32], [32])
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64])
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7*7*64, 1024], [1024])

        # apply dropout
        fc_1_drop = tf.nn.dropout(fc_1, keep_prob)

    with tf.variable_scope("output"):
        _output = layer(fc_1_drop, [1024, 10], [10])

    return _output


def loss(output, y):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy)
    return loss


def training(cost, learning_rate, global_step):
    print("learning_rate is %r" % learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cost, global_step=global_step)
    return train_step


def evaluate(output, y):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def save_output(final_output):
    a = mnist.test.images
    b = mnist.test.ids
    out_seq = []
    for i in range(5):
        start = i * 10000
        end = (i + 1) * 10000
        eval_feed = {x: a[start:end], keep_prob: 1.0}
        rrr = sess.run(final_output, feed_dict=eval_feed)
        DAT = np.column_stack((np.array(b[start:end]), rrr))
        out_seq.append(DAT)
    outfile = np.row_stack(out_seq)
    np.savetxt("/tmp/sample.csv", outfile, delimiter=",", fmt="%s")
    print ("/tmp/sample.csv is saved")

if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32) # dropout probability

    output = inference(x, keep_prob)
    cost = loss(output, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = training(cost, learning_rate, global_step)
    eval_op = evaluate(output, y_)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()

    sess.run(init_op)
    saver = tf.train.Saver()
    restore_path = "/tmp/will/20/model_conv.ckpt"
    saver.restore(sess, restore_path)
    print ("read file from %s " % restore_path)

    for i in range(10):
        print("epoch", i)
        total_batch = int(mnist.train.num_examples / batch_size)
        for j in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.4})
            '''validation'''
            # result = sess.run([eval_op, cost], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            # accuracy = result[0]
            # my_loss = result[1]
            # print("Accuracy: ", accuracy)
            # print("loss: ", my_loss)
        print("---------------")
        for j in range(5):
            start = j * 10000
            end = (j + 1) * 10000
            eval_feed = {x: mnist.test.images[start:end], y_: mnist.test.labels[start:end], keep_prob: 1.0}
            result = sess.run([eval_op, cost], feed_dict=eval_feed)
            print("acc : %s, loss: %s" % (result[0], result[1]))
        print("---------------")

    save_path = saver.save(sess, "/tmp/will/model_conv.ckpt")
    print("Model saved in file: %s" % save_path)
    save_output(output)
    sess.close()
