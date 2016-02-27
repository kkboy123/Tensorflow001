__author__ = 'kkboy'

import tensorflow as tf

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print ("x + y = %s" % sess.run(x + y))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

minus = tf.sub(a, b)
mul = tf.mul(a, b)
matmul1 = tf.matmul(a, b)
matmul2 = tf.matmul(b, a)

with tf.Session() as sess:
    print "run minus operation"
    print sess.run(minus, feed_dict={a: 5, b: 2})
    print "run mul operation"
    print sess.run(mul, feed_dict={a: 5, b: 2})
    print "run matmul1 operation"

    '''  a is a 2x1 matrix
         a : [[3.0, 3.0]]
         b is a 1x2 matrix
         b : [[2.0],
              [2.0]]
    '''
    print sess.run(matmul1, feed_dict={a: [[3., 3.]],
                                       b: [[2.], [2.]]})
    print "run matmul2 operation"
    print sess.run(matmul2, feed_dict={a: [[3., 3.]],
                                       b: [[2.], [2.]]})
