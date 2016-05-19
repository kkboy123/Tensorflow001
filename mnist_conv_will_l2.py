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
base_learning_rate = 1e-2
load_model = False

'''
Create the model
x, which is a 784-dim vector, is input layer
W, which is a 784x10 matrix, is the weight of the input layer
b, which is a 784-dim vector, is the bias of the input layer
y, which is a one-hot vector (10-dim), is the prediction
'''


def layer(_input, weight_shape, bias_shape, name="default_name"):
    weight_init = tf.random_normal_initializer(stddev=(2.0 / weight_shape[0])**0.5)
    bias_init = tf.constant_initializer(value=0)
    w = tf.get_variable(name + "_W", weight_shape, initializer=weight_init)
    b = tf.get_variable(name + "_b", bias_shape, initializer=bias_init)
    tf.add_to_collection('L2_losses', tf.nn.l2_loss(w))
    tf.add_to_collection('L2_losses', tf.nn.l2_loss(b))
    return tf.nn.relu(tf.matmul(_input, w) + b)


def conv2d(_input, weight_shape, bias_shape, name="default_name"):
    incoming = weight_shape[0] * weight_shape[1] * weight_shape[2]
    weight_init = tf.random_normal_initializer(stddev=(2.0 / incoming)**0.5)
    w = tf.get_variable(name + "_W", weight_shape, initializer=weight_init)
    bias_init = tf.constant_initializer(value=0)
    b = tf.get_variable(name + "_b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input, w, strides=[1, 1, 1, 1], padding='SAME'), b))


def max_pool(_input, k=2):
    return tf.nn.max_pool(_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def inference(_x, _keep_prob):
    _x = tf.reshape(_x, shape=[-1, 28, 28, 1])
    with tf.variable_scope("conv_1"):
        conv_1 = conv2d(_x, [5, 5, 1, 32], [32], name="conv1")
        pool_1 = max_pool(conv_1)

    with tf.variable_scope("conv_2"):
        conv_2 = conv2d(pool_1, [5, 5, 32, 64], [64], name="conv2")
        pool_2 = max_pool(conv_2)

    with tf.variable_scope("fc"):
        pool_2_flat = tf.reshape(pool_2, [-1, 7 * 7 * 64])
        fc_1 = layer(pool_2_flat, [7 * 7 * 64, 1024], [1024], name="fc1")
        '''apply dropout'''
        fc_1_drop = tf.nn.dropout(fc_1, _keep_prob, name="dropout1")

    with tf.variable_scope("output"):
        _output = layer(fc_1_drop, [1024, 10], [10], name="fc2")

    return _output


def loss(_output, _labels):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(_output, _labels)
    _logloss = tf.reduce_mean(xentropy)
    # y_conv = tf.nn.softmax(_output)
    # panalty = -tf.reduce_sum((1 - y_)*tf.log(1 - tf.clip_by_value(y_conv, 1e-25, 1.0)), 1)
    # # panalty = tf.nn.softmax_cross_entropy_with_logits(1 - _output, 1 - _labels)
    # _panalty_logloss = tf.reduce_mean(panalty)
    _total_l2_loss = tf.add_n(tf.get_collection('L2_losses'), name='total_l2_loss')
    _loss = _logloss + (5e-4 * _total_l2_loss)  # + _panalty_logloss
    return _loss, _logloss, _total_l2_loss  # , _panalty_logloss


def training(_cost, _learning_rate, _global_step):
    print("learning_rate is %r" % _learning_rate)
    learning_rate = tf.train.exponential_decay(
        _learning_rate,     # Base learning rate.
        _global_step,       # Current index into the dataset.
        10000,              # Decay step.
        0.95,               # Decay rate.
        staircase=True)
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    _train_step = optimizer.minimize(_cost, global_step=_global_step)

    return _train_step


def evaluate(_output, _labels):
    correct_prediction = tf.equal(tf.argmax(_output, 1), tf.argmax(_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy


def save_output(final_output):
    a = mnist.test.images
    b = mnist.test.ids
    out_seq = []
    for _i in range(5):
        _start = _i * 10000
        _end = (_i + 1) * 10000
        _eval_feed = {x: a[_start:_end], keep_prob: 1.0}
        rrr = sess.run(tf.nn.softmax(final_output), feed_dict=_eval_feed)
        dat = np.column_stack((np.array(b[_start:_end]), rrr))
        out_seq.append(dat)
    outfile = np.row_stack(out_seq)
    np.savetxt("/tmp/sample.csv", outfile, delimiter=",", fmt="%s")
    print ("/tmp/sample.csv is saved")

if __name__ == '__main__':

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    output = inference(x, keep_prob)
    cost, logloss, l2_loss = loss(output, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = training(cost, base_learning_rate, global_step)
    eval_op = evaluate(output, y_)
    sess = tf.Session()
    init_op = tf.initialize_all_variables()

    sess.run(init_op)
    saver = tf.train.Saver()
    load_model = True
    if load_model:
        restore_path = "/tmp/will/model_conv.ckpt"
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
        total_loss = 0
        print("---------------")
        for j in range(5):
            start = j * 10000
            end = (j + 1) * 10000
            eval_feed = {x: mnist.test.images[start:end], y_: mnist.test.labels[start:end], keep_prob: 1.0}
            result = sess.run([eval_op, cost, logloss, l2_loss], feed_dict=eval_feed)
            print("acc : %s, cost: %s, logloss: %s, l2loss: %s" %
                  (result[0], result[1], result[2], result[3]))
            total_loss += result[2]
        print("logloss : %s" % (total_loss / 5))
        print("---------------")

    save_path = saver.save(sess, "/tmp/will/model_conv.ckpt")
    print("Model saved in file: %s" % save_path)
    save_output(output)
    sess.close()
