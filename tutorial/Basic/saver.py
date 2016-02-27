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
# Add ops to save and restore all the variables.
saver = tf.train.Saver()
# Save the variables to disk.
save_path = saver.save(sess, "/tmp/model.ckpt")
print("Model saved in file: %s" % save_path)
