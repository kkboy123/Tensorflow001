import tensorflow as tf
import numpy as np

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but Tensorflow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0), name='weight_W')
b = tf.Variable(tf.zeros([1]), name='bias_b')

with tf.name_scope("Wx_b") as scope:
    y = W * x_data + b

# Add summary ops to collect data
W_hist = tf.histogram_summary("weights_summary", W)
b_hist = tf.histogram_summary("biases_summary", b)
y_hist = tf.histogram_summary("y_summary", y)

# Minimize the mean squared errors.
with tf.name_scope("xent") as scope:
    loss = tf.reduce_mean(tf.square(y - y_data))
    loss_summ = tf.scalar_summary("reduce_mean", loss)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

# Merge all the summaries and write them out to /tmp/mnist_logs
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/helloBoard0_g", sess.graph_def)
init = tf.initialize_all_variables()

# Launch the graph
sess.run(init)

# Fit the line.
for step in xrange(201):
    sess.run(train)
    if step % 20 == 0:
        result = sess.run([merged])
        summary_str = result[0]
        writer.add_summary(summary_str, step)
        print("W and b at step %s: (%s, %s)" % (step, sess.run(W), sess.run(b)))

sess.close()
# Learns best fit is W: [0.1], b: [0.3]