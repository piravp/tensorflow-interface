import tensorflow as tf


##########################################################################################
                            # ***** Cost functions ******
##########################################################################################

mse = lambda target, output: tf.losses.mean_squared_error(target, output)
sigmoid = lambda target, output: tf.losses.sigmoid_cross_entropy(target, output)
softmax_cross_entropy = lambda target, output : tf.losses.softmax_cross_entropy(target, output)

