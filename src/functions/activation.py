import tensorflow as tf

##########################################################################################
                            # **** Activation functions ****
##########################################################################################

relu = lambda inputs, name : tf.nn.relu(inputs, name=name + '-relu')
softmax = lambda inputs, name : tf.nn.softmax(inputs, name=name + '-softmax')
