import tensorflow as tf
import numpy as np

# A general ann module = a layer of neurons (the output) plus its incoming weights and biases.
class GANNModule():
    def __init__(self, ann, index, invariable, insize, outsize, weight_range, activation_func, layertype):
        self.ann = ann                                      # ann
        self.index = index
        self.input = invariable                             # Either the gann's input variable or the upstream module's output
        self.insize = insize                                # Number of neurons feeding into this module
        self.size = outsize                                 # Number of neurons in this module
        self.weight_range = weight_range
        self.activation_function = activation_func
        self.name = "Module-"+str(self.index) + layertype   # Layertype = 'input', 'hidden' or 'output'
        self.build()

    def build(self):
        # mona = self.name
        n = self.size
        self.weights = tf.Variable(np.random.uniform(self.weight_range[0], self.weight_range[1], size=(self.insize,n)), name=self.name+'-weights',trainable=True)   # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(self.weight_range[0], self.weight_range[1], size=n), name=self.name+'-biases', trainable=True)                  # First bias vector
        self.output =  self.activation_function(tf.matmul(self.input,self.weights)+self.biases,name=self.name)           # tf.nn.relu()
        self.ann.add_layer(self)
