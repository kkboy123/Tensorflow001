from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from parser import read_data_sets
from destort import get_generate
import tensorflow as tf
import numpy as np


def weight_variable(shape, name='None'):
    input_size = 1
    for _i in shape[:-1]:
        input_size *= _i
    # initial = tf.truncated_normal(shape, stddev=0.05, name=name)
    initial = tf.truncated_normal(shape, stddev=(2.0 / input_size)**0.50, name=name)
    return tf.Variable(initial)


def bias_variable(shape, name='None'):
    initial = tf.truncated_normal(mean=0.092, stddev=0.002, shape=shape, name=name)
    return tf.Variable(initial,)


def conv2d(_x, _w):
    return tf.nn.conv2d(_x, _w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(_x):
    return tf.nn.max_pool(_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = read_data_sets("data2", one_hot=True)
mnist.generate = get_generate("data2")
sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
c1_input = 64
c2_input = 64
fc1_input = 128
output_class = 10
base_learning_rate = 1e-4
epsilon = 1e-3
batch_size = 100
epoch_size = 200
start_from = 50

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


# dropout for reducing over-fitting
with tf.name_scope("dropout") as scope:
    keep_prob0 = tf.placeholder("float")
    h_pool2_drop = tf.nn.dropout(h_pool2, keep_prob0)


# Fully (densely) connected Layer
with tf.name_scope("fc1") as scope:
    W_fc1 = weight_variable([7 * 7 * c2_input, fc1_input], name='W_fc1')
    tf.add_to_collection('L2_loss', tf.nn.l2_loss(W_fc1))
    h_pool2_flat = tf.reshape(h_pool2_drop, [-1, 7*7*c2_input])
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

# readout layer
with tf.name_scope("sftmax") as scope:
    W_fc2 = weight_variable([fc1_input, output_class], name='W_fc2')
    tf.add_to_collection('L2_loss', tf.nn.l2_loss(W_fc2))
    b_fc2 = bias_variable([output_class], name='b_fc2')
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
with tf.name_scope("xent") as scope:
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    # logloss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)) + (1 - y_) * tf.log(tf.clip_by_value(1 - y_conv, 1e-25, 1.0)))
    logloss = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)) +\
                                            (1 - y_) * tf.log(tf.clip_by_value(1 - y_conv, 1e-25, 1.0)), 1))
    # logloss2 = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)), 1))
    logloss2 = -tf.reduce_mean(tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-25, 1.0)), 1))
    # logloss2 = (tf.reduce_sum(y_ * y_conv, 1))
    # mean, var = tf.nn.moments(logloss2)
    l2_loss = tf.add_n(tf.get_collection('L2_loss'), name='total_l2_loss')
    cost = logloss  #+ (5e-4 * l2_loss)

with tf.name_scope("train") as scope:
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_step = tf.train.AdamOptimizer(base_learning_rate).minimize(cost, global_step=global_step)
    # train_step = tf.train.AdagradOptimizer(1e-4).minimize(logloss)

# Train and test the model
with tf.name_scope("test") as scope:
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # accuracy_summary = tf.scalar_summary("accuracy", accuracy)


def my_eval(_feed):
    _result = sess.run([accuracy, logloss, logloss2], feed_dict=_feed)
    _acc = _result[0]
    _loss = _result[1]
    _loss2 = _result[2]
    # _total_cost = _result[3]
    print("Accuracy : %s; loss : %s; logloss: %s" % (_acc, _loss, _loss2))
    return _loss2


def total_eval(validation_size, epoch_index=0):
    print ("--------------------------------")
    total_logloss = 0.0
    for _i in range(5):
        start = _i * 10000
        end = start + min(10000, validation_size)
        eval_feed = {x: mnist.test.images[start:end], y_: mnist.test.labels[start:end], keep_prob: 1.0, keep_prob0: 1.0}
        train_logloss = my_eval(eval_feed)
        total_logloss += train_logloss
    total_logloss /= 5.0
    print ("total logloss : %s" % total_logloss)
    print ("--------------------------------")
    if total_logloss < 0.034 and validation_size == 10000:
        save_output(epoch_index)
    return total_logloss


def save_output(_i=0):
    a = mnist.test.images
    b = mnist.test.ids
    out_seq = []
    for i in range(5):
        start = i * 10000
        end = (i + 1) * 10000
        eval_feed = {x: a[start:end], keep_prob: 1.0, keep_prob0: 1.0}
        rrr = sess.run(y_conv, feed_dict=eval_feed)
        DAT = np.column_stack((np.array(b[start:end]), rrr))
        out_seq.append(DAT)
    outfile = np.row_stack(out_seq)
    filename = "/Users/will/HelloPycharm/gen_data/sample{0}.csv".format(_i)
    np.savetxt(filename, outfile, delimiter=",", fmt="%s")
    print ("%s is saved" % filename)


init = tf.initialize_all_variables()

''' Launch the graph'''
sess.run(init)

''' Add ops to save and restore all the variables.'''
saver = tf.train.Saver()
restore_path = "/Users/will/HelloPycharm/gen_data/model.ckpt-49"
saver.restore(sess, restore_path)
print ("read file from %s " % restore_path)

for i in range(start_from, epoch_size):
    print("epoch %s" % i)
    # batch = mnist.train.next_batch(50)
    for j in range(int(mnist.generate.num_examples/batch_size)):
        batch = mnist.generate.next_batch(batch_size)
        if j % 550 == 549:
            feed = {x: batch[0], y_: batch[1], keep_prob: 1.0, keep_prob0: 1.0}
            my_eval(feed)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.35, keep_prob0: 0.50})
    if i % 10 == 9:
        total_eval(10000, i)
        save_path = saver.save(sess, "/Users/will/HelloPycharm/gen_data/model.ckpt", global_step=i)
        print("Model saved in file: %s" % save_path)
    else:
        valid_loss = total_eval(100)
        if valid_loss < 0.056:
            save_path = saver.save(sess, "/Users/will/HelloPycharm/gen_data/model.ckpt", global_step=i)
            print("Model saved in file: %s" % save_path)
            save_output(i)


'''Save the variables to disk.'''
# save_path = saver.save(sess, "/tmp/mnist_conv_lrn_logs/model.ckpt")

'''Save sample.csv'''
# save_output()
sess.close()


