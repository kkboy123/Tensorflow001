__author__ = "kkboy"

import tensorflow as tf

hello = tf.constant('hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)
