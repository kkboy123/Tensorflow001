__author__ = "kkboy"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True)

a = mnist.test.images[0]
b = ["@" if x>0 else "'" for x in a ]
for i in range(28):
    print b[i*28:(i+1)*28]
