__author__ = 'kkboy'

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn


''' Defining some hyper-params'''
# this is the parameter for input_size in the basic LSTM cell
num_units = n_hidden = 40
# num_units and input_size will be the same
input_size = n_input = 2
batch_size = 10
seq_len = n_steps = 5
num_epochs = 1000
n_classes = 1

''' tf Graph input'''
x = tf.placeholder("float", [batch_size, n_steps, n_input])
''' Tensorflow LSTM cell requires 2x n_hidden length (state & cell)'''
istate = tf.placeholder("float", [batch_size, 2*n_hidden])
y = tf.placeholder("float", [batch_size, n_classes])

''' Define weights'''
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def gen_data(min_length=50, max_length=55, n_batch=50):
    _X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                        axis=-1)
    _y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(2, min_length)
        # length = np.random.randint(min_length, max_length)

        # Zero out _X after the end of the sequence
        _X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        _X[n, np.random.randint(abs(length / 2 - 1), length), 1] = 1
        _X[n, np.random.randint(length / 2, length), 1] = 1
        # Multiply and sum the dimensions of _X to get the target value
        _y[n] = np.sum(_X[n, :, 0] * _X[n, :, 1])
    return _X, _y


def myRNN(_x, _istate, _weights, _biases):
    ''' input shape: (batch_size, n_steps, n_input) '''
    _x = tf.transpose(_x, [1, 0, 2])  # permute n_steps and batch_size
    ''' _x: (n_steps,batch_size, n_input) '''
    ''' Reshape to prepare input to hidden activation '''
    _x = tf.reshape(_x, [-1, input_size]) # (n_steps*batch_size, n_input)
    ''' Linear activation '''
    _x = tf.matmul(_x, _weights['hidden']) + _biases['hidden'] # (n_steps*batch_size, n_hidden)
    ''' Define a lstm cell with tensorflow '''
    # num units is the input-size for this cell
    lstm_cell = rnn_cell.BasicLSTMCell(num_units)
    ''' Split data because rnn cell needs a list of inputs for the RNN inner loop '''
    _x = tf.split(0, seq_len, _x) # n_steps * (batch_size, n_hidden)
    ''' Get lstm cell output '''
    outputs, states = rnn.rnn(lstm_cell, _x, initial_state=_istate)
    #outputs :  n_steps * (batch_size, n_hidden)
    '''
    Linear activation
    Get inner loop last output
    '''
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']

pred = myRNN(x, istate, weights, biases)

''' Define loss and optimizer'''
cost = tf.reduce_mean(tf.pow(pred - y, 2))
train_op = tf.train.AdagradOptimizer(0.005).minimize(cost)

''' Generate Validation Data '''
test_dataX, tempY = gen_data(seq_len, seq_len, batch_size)
test_dataY = [[i] for i in tempY]

''' Execute '''
with tf.Session() as sess:
    # initialize all variables in the model
    tf.initialize_all_variables().run()
    for k in range(num_epochs):
        # Generate Data for each epoch
        # What this does is it creates a list of of elements of length seq_len, each of size [batch_size,input_size]
        # this is required to feed data into rnn.rnn
        batch_xs, tempys = gen_data(seq_len, seq_len, batch_size)
        batch_ys = [[tempy] for tempy in tempys]
        sess.run(train_op, feed_dict={x: batch_xs,
                                      y: batch_ys,
                                      istate: np.zeros((batch_size, 2*n_hidden))})
        result = sess.run([pred, cost], feed_dict={x: test_dataX,
                                                   y: test_dataY,
                                                   istate: np.zeros((batch_size, 2*n_hidden))})
        print "Validation cost: {}, on Epoch {}".format(result[1], k)
    print "outputs3 {}".format(result[0])
print test_dataY
