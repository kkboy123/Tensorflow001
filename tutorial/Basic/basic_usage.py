__author__ = 'kkboy'

import tensorflow as tf

# Create two variables.
weights = tf.Variable(tf.random_normal([5, 2], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([2]), name="biases")

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Later, when launching the model
sess = tf.Session()
# Run the init operation.
sess.run(init_op)
# Use the model
print 'weights'
print sess.run(weights)
print 'biases'
print sess.run(biases)

# Create another variable with the same value as 'weights'.
w2 = tf.Variable(weights.initialized_value(), name="w2")
# Create another variable with twice the value of 'weights'
w_twice = tf.Variable(weights.initialized_value() * 2.0, name="w_twice")
sess.run(tf.initialize_all_variables())
results = sess.run([w2, w_twice])
print 'w2'
print results[0]
print 'w_twice'
print results[1]
