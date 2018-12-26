import tensorflow as tf
import numpy as np
import src.helpers.tflowtools as TFT
from src.Layer import GANNModule
import os
from src.helpers.Visualizer import Visualizer

# Turn off TF warnings about AVX etc.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ANN:
    def __init__(self, cman, dimensions, hidden_activation, output_activation, cost_func,
                 lrate, initial_weight_range, optimizer, mb_size,
                 map_batch_size, steps, val_interval, map_layers, map_dendro, display_weights, display_biases):

        print("\n~~~~~~~~~~~~~~~~~~~~~~ Initialization ~~~~~~~~~~~~~~~~~~~~~")
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
        print('Building network...')

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
        # Split dataset
        inputs, targets = np.asarray(cases).T
        feeder = {self.input: inputs.tolist(), self.target: targets.tolist()}
        predictions, error = self.session.run([self.output, self.error], feeder)

        # Output node with highest activation level correspond to target value
        top_k = self.session.run(tf.nn.in_top_k(predictions, TFT.one_hot_vectors_to_ints(one_hot_vectors=targets), 1))

        return 100.0*np.sum(top_k)/len(inputs), error

    def validation_testing(self, step):
        validation_cases = self.caseman.get_validation_cases()
        accuracy, error = self.compute_accuracy(validation_cases)
        self.validation_step.append(step)
        self.validation_error.append(error)
        val_i = int(step/self.validation_interval)
        val_steps = int(self.steps / self.validation_interval)
        print("Accuracy on validation set [{0}/{1}]: {2:.3f} %".format(val_i, val_steps, accuracy))

    def train(self):
        print('Initializing training...')

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.training_step, self.training_error = [], []
        self.validation_step, self.validation_error = [], []

        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~")
        # Training
        for step in range(1, self.steps + 1):
            inputs, targets = self.caseman.get_minibatch(self.minibatch_size)
            # targets = targets.T
            feeder = {self.input: inputs.tolist(), self.target: targets.tolist()}
            _, error = self.session.run([self.trainer, self.error], feed_dict=feeder)
            self.training_step.append(step)
            self.training_error.append(error)

            # Validation
            if step % self.validation_interval == 0:
                self.validation_testing(step=step)

    def test(self):
        print("\n~~~~~~~~~~~~~~~~~~~~~~~~~ Testing ~~~~~~~~~~~~~~~~~~~~~~~~")
        train_accuracy, _ = self.compute_accuracy(self.caseman.training_cases)
        test_accuracy, _ = self.compute_accuracy(self.caseman.testing_cases)
        print('Train accuracy: {0:.3f} %'.format(train_accuracy))
        print('Test accuracy: {0:.3f} %'.format(test_accuracy))


    # Add layer(GANNModule) in Layer.py's build()
    def add_layer(self, layer):
        self.layers.append(layer)


    def run(self):
        # Build network
        self.build()

        # Training
        self.train()

        # Testing
        self.test()

        # Visualization
        visual_options = {
            'casemanager': self.caseman,
            'session': self.session,
            'input': self.input,
            'map_layers': self.map_layers,
            'map_dendrograms': self.map_dendrograms,
            'display_weights': self.display_weights,
            'display_biases': self.display_biases,
            'map_batch_size': self.map_batch_size,
            'layers': self.layers,
            'training_step': self.training_step,
            'training_error': self.training_error,
            'validation_step': self.validation_step,
            'validation_error': self.validation_error
        }
        visualizer = Visualizer(**visual_options)
        visualizer.visualize()


