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
c1_input = 32
c2_input = 64
fc1_input = 1024
fc2_input = 512
output_class = 10
epsilon = 1e-3

# reshape x for new W
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first convolutional layer l
with tf.name_scope("conv1") as scope:
    W_conv1 = weight_variable([5, 5, 1, c1_input], name='W_conv1')
    h_conv1 = conv2d(x_image, W_conv1)
    batch_mean_c1, batch_var_c1 = tf.nn.moments(h_conv1, [0, 1, 2])
    h_conv1_hat = (h_conv1 - batch_mean_c1) / tf.sqrt(batch_var_c1 + epsilon)
    # create two new parameters, scale and beta (shift)
    scale_c1 = tf.Variable(tf.ones([c1_input]))
    beta_c1 = tf.Variable(tf.zeros([c1_input]))
    bn_conv1 =tf.nn.relu(scale_c1 * h_conv1_hat + beta_c1)
    h_pool1 = max_pool_2x2(bn_conv1)
    # h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

# second convolutional layer
with tf.name_scope("conv2") as scope:
    W_conv2 = weight_variable([5, 5, c1_input, c2_input], name='W_conv2')
    h_conv2 = conv2d(h_pool1, W_conv2)
    batch_mean_c2, batch_var_c2 = tf.nn.moments(h_conv2, [0, 1, 2])
    h_conv2_hat = (h_conv2 - batch_mean_c2) / tf.sqrt(batch_var_c2 + epsilon)
    # create two new parameters, scale and beta (shift)
    scale_c2 = tf.Variable(tf.ones([c2_input]))
    beta_c2 = tf.Variable(tf.zeros([c2_input]))
    bn_conv2 =tf.nn.relu(scale_c2 * h_conv2_hat + beta_c2)
    h_pool2 = max_pool_2x2(bn_conv2)

# Fully (densely) connected Layer
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([7 * 7 * c2_input, fc1_input], name='W_fc1')
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*c2_input])
    h_fc1_mat = tf.matmul(h_pool2_flat, W_fc1)
    batch_mean_fc1, batch_var_fc1 = tf.nn.moments(h_fc1_mat, [0])
    h_fc1_hat = (h_fc1_mat - batch_mean_fc1) / tf.sqrt(batch_var_fc1 + epsilon)
    scale_fc1 = tf.Variable(tf.ones([fc1_input]))
    beta_fc1 = tf.Variable(tf.zeros([fc1_input]))
    h_fc1 = tf.nn.relu(scale_fc1 * h_fc1_hat + beta_fc1)
    # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout for reducing over-fitting
with tf.name_scope("dropout") as scope:
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

''' Fully (densely) connected Layer fc2'''
with tf.name_scope("fc2") as scope:
    W_fc2 = weight_variable([fc1_input, fc2_input], name='W_fc2')
    h_fc2_mat = tf.matmul(h_fc1_drop, W_fc2)
    batch_mean_fc2, batch_var_fc2 = tf.nn.moments(h_fc2_mat, [0])
    h_fc2_hat = (h_fc2_mat - batch_mean_fc2) / tf.sqrt(batch_var_fc2 + epsilon)
    '''create two new parameters, scale and beta (shift)'''
    scale_fc2 = tf.Variable(tf.ones([fc2_input]))
    beta_fc2 = tf.Variable(tf.zeros([fc2_input]))
    h_fc2 = tf.nn.relu(scale_fc2 * h_fc2_hat + beta_fc2)


''' dropout for reducing over-fitting dropout2'''
with tf.name_scope("dropout") as scope:
    keep_prob2 = tf.placeholder("float")
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob2)


''' readout layer'''
with tf.name_scope("softmax") as scope:
    W_fc3 = weight_variable([fc2_input, output_class], name='W_fc3')
    b_fc3 = bias_variable([output_class], name='b_fc3')
    y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

# Define loss and optimizer
with tf.name_scope("xent") as scope:
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    logloss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y_conv, 1e-25, 1.0)))
    # logloss = -tf.reduce_sum(y_*tf.log(y_conv))
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


def total_eval():
    test_result = sess.run([logloss2, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, keep_prob2: 1.0})
    print ("--------------------------------")
    print ("Total logloss of testing data : %s" % test_result[0])
    '''Print the final accuracy'''
    print ("test accuracy %g" % test_result[1])
    total_logloss = 0.0
    for i in range(5):
        start = i * 10000
        end = (i + 1) * 10000
        eval_feed = {x: mnist.train.images[start:end], y_: mnist.train.labels[start:end], keep_prob: 1.0, keep_prob2: 1.0}
        print ("test accuracy %g" % accuracy.eval(feed_dict=eval_feed))
        train_logloss = sess.run(logloss2, feed_dict=eval_feed)
        print ("logloss : %s" % train_logloss)
        total_logloss += train_logloss
    total_logloss /= 5.0
    print ("total logloss : %s" % total_logloss)
    print ("--------------------------------")


init = tf.initialize_all_variables()
''' Launch the graph'''
sess.run(init)
''' Add ops to save and restore all the variables.'''
saver = tf.train.Saver()

for i in range(6001):
    batch = mnist.test.next_batch(50)
    if i % 100 == 0:
        feed = {x: batch[0], y_: batch[1], keep_prob: 1.0, keep_prob2: 1.0}
        result = sess.run([accuracy, logloss, logloss2, y_conv], feed_dict=feed)
        acc = result[0]
        loss = result[1]
        # writer.add_summary(summary_str, i)
        print("Accuracy at step %s: %s; loss : %s" % (i, acc, loss))
        if np.isnan(loss):
            print("logloss2 : %s" % result[2])
            print("y_conv : %s" % result[3])
            break
    if i % 500 == 0 and i > 0:
        total_eval()
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, keep_prob2: 0.4})


''' Save the variables to disk.'''
save_path = saver.save(sess, "/tmp/mnist_conv_lrn_logs/model.ckpt")
print("Model saved in file: %s" % save_path)
sess.close()
