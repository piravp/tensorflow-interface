import tensorflow as tf
import numpy as np
import src.helpers.tflowtools as TFT
from src.Layer import GANNModule
import os

# Turn off TF warnings about AVX etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ANN:
    def __init__(self, cman, dimensions, hidden_activation, output_activation, cost_func,
                 lrate, initial_weight_range, optimizer, mb_size,
                 map_batch_size, steps, val_interval, map_layers, map_dendro, display_weights, display_biases):


        print('Initializing neural network...')
        self.caseman = cman
        dimensions.insert(0, cman.n_features)
        dimensions.append(cman.n_classes)
        self.network_dimensions = dimensions
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.cost_function = cost_func
        self.learning_rate = lrate
        self.initial_weight_range = initial_weight_range
        self.optimizer = optimizer
        self.minibatch_size = mb_size
        self.map_batch_size = map_batch_size
        self.validation_interval = val_interval
        self.steps = steps
        self.map_layers = map_layers
        self.map_dendrograms = map_dendro
        self.display_weights = display_weights
        self.display_biases = display_biases

        self.layers = []

    # Build graph
    def build(self):
        # This is essential for doing multiple runs!!
        tf.reset_default_graph()

        # Build input layer of the network
        input_size = self.network_dimensions[0]
        self.input = tf.placeholder(tf.float64, shape=(None, input_size), name='Input')

        # Init variables
        gann_module = None
        prev_output = self.input
        prev_output_size = input_size

        # Build hidden modules (=neurons + incoming bias/weights)
        for index, output_dimension in enumerate(self.network_dimensions[1:]):
            gann_module      = GANNModule(self, index, prev_output, prev_output_size, output_dimension, self.initial_weight_range, self.hidden_activation, '-hidden')
            prev_output      = gann_module.output
            prev_output_size = gann_module.size

        # bør kunne erstatte gann_module.output/gann_module.size med prev_output/prev_output_size
        self.output = GANNModule(self, -1, gann_module.output, gann_module.size, self.network_dimensions[-1], self.initial_weight_range, self.output_activation, '-output')
        # final prediction
        self.output = self.layers[-1].output  # Cast from GANNModule to Tensor
        self.target = tf.placeholder(tf.float64, shape=(None, gann_module.size),name='Target')
        self.configure_learning()

    def configure_learning(self):
        self.error = self.cost_function(self.target, self.output)
        optimizer = self.optimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error)   #Backpropagation

    def compute_accuracy(self, cases):
        inputs, targets = np.asarray(cases).T
        # Caseman.show_image(inputs[0], targets[0])
        feeder = {self.input: inputs.tolist(), self.target: targets.tolist()}
        predictions, error = self.session.run([self.output, self.error], feeder)

        # # Test if output node with highest activation level correspond to target value
        # target_type = self.casemanager.casefunc.__name__
        # if target_type == 'one_hot' or target_type == 'ninja':
        top_k = self.session.run(tf.nn.in_top_k(predictions, TFT.one_hot_vectors_to_ints(one_hot_vectors=targets), 1))
        # elif target_type == 'autoencoder':
        # top_k = self.session.run(tf.nn.in_top_k(predictions, TFT.bitdataset_to_intlist(matrix=targets), 1))
        # else:
        # top_k = self.session.run(tf.nn.in_top_k(predictions, targets, 1))
        return 100.0*np.sum(top_k)/len(inputs), error

    def validation_testing(self, step):
        if step % self.validation_interval == 0:
            validation_cases = self.caseman.get_validation_cases()
            accuracy, error = self.compute_accuracy(validation_cases)
            self.validation_step.append(step)
            self.validation_error.append(error)
            print("Accuracy on validation set after {0}th minibatch: {1:.3f} %".format(step, accuracy))

    def train(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.training_step, self.training_error = [], []
        self.validation_step, self.validation_error = [], []

        # Training
        for step in range(1, self.steps + 1):
            inputs, targets = self.caseman.get_minibatch(self.minibatch_size)
            # targets = targets.T
            feeder = {self.input: inputs.tolist(), self.target: targets.tolist()}
            _, error = self.session.run([self.trainer, self.error], feed_dict=feeder)
            self.training_step.append(step)
            self.training_error.append(error)

            # Validation
            self.validation_testing(step=step)


    # Add layer(GANNModule) in Layer.py's build()
    def add_layer(self, layer):
        self.layers.append(layer)


    def run(self):
        # Build network
        self.build()

        # Training
        self.train()

