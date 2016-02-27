__author__ = 'kkboy'

import tensorflow as tf

# Create two variables.
weights = tf.Variable(tf.random_normal([5, 2], stddev=0.35), name="weights")
biases = tf.Variable(tf.zeros([2]), name="biases")

# Add an op to initialize the variables.
init_op = tf.initialize_all_variables()
# Open an interavtive session
sess = tf.InteractiveSession()
# Instead of running the init operation.
# Initial the variable you want
weights.initializer.run()
biases.initializer.run()

print 'weights'
print weights.eval()
print 'biases'
print biases.eval()