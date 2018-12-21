import tensorflow as tf


##########################################################################################
                        # ***** Optimizer functions ******
##########################################################################################

adagrad = lambda lrate : tf.train.AdagradOptimizer(learning_rate=lrate, name="adagrad")
adam = lambda lrate : tf.train.AdamOptimizer(learning_rate=lrate, name = "adam")
gradient_descent = lambda lrate : tf.train.GradientDescentOptimizer(learning_rate=lrate, name = "gradient_descent")
rmsprop = lambda lrate : tf.train.RMSPropOptimizer(learning_rate=lrate, name = "rmsprop")
