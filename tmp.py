with tf.name_scope("conv3") as scope:
    W_conv3 = weight_variable([5, 5, c2_input, c3_input], name='W_conv3')
    h_conv3 = conv2d(h_pool2, W_conv3)
    batch_mean_c3, batch_var_c3 = tf.nn.moments(h_conv3, [0, 1, 2])
    h_conv3_hat = (h_conv3 - batch_mean_c3) / tf.sqrt(batch_var_c3 + epsilon)
    # create two new parameters, scale and beta (shift)
    scale_c3 = tf.Variable(tf.ones([c3_input]))
    beta_c3 = tf.Variable(tf.zeros([c3_input]))
    bn_conv3 =tf.nn.relu(scale_c3 * h_conv3_hat + beta_c3)
    h_pool3 = bn_conv3
    # h_pool3 = max_pool_3x3(bn_conv3)