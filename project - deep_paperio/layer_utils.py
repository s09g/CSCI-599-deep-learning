import tensorflow as tf


def conv2d(x_tensor, conv_num_outputs, conv_ksize=3, conv_stride = 1):
    _, input_width, input_height, input_depth = x_tensor.get_shape().as_list()

    weights = tf.Variable(tf.truncated_normal([conv_ksize, conv_ksize, input_depth, conv_num_outputs],
                                              mean=0.0, stddev=0.05, dtype=tf.float32))
    biases = tf.Variable(tf.zeros(conv_num_outputs), dtype=tf.float32)

    conv = tf.nn.conv2d(input=x_tensor, filter=weights, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, biases)
    conv = tf.nn.relu(conv)
    return conv


def maxpool(x_tensor):
    return tf.nn.max_pool(x_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def flatten(x_tensor):
    return tf.contrib.layers.flatten(x_tensor)


def fully_conn(x_tensor, num_outputs):
    return tf.contrib.layers.fully_connected(inputs = x_tensor, num_outputs=num_outputs)


def output(x_tensor, num_outputs):
    return tf.contrib.layers.fully_connected(inputs=x_tensor, num_outputs=num_outputs, activation_fn=None)

