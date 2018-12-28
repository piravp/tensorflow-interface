import src.helpers.tflowtools as TFT
import matplotlib.pyplot as plt
import numpy as np

class Visualizer():
    def __init__(self, casemanager, session, input, map_layers, map_dendrograms, display_weights, display_biases, map_batch_size, layers,
                 training_step, training_error, validation_step, validation_error):
        self.casemanager = casemanager
        self.session = session
        self.input = input
        self.map_batch_size = map_batch_size
        self.map_layers = map_layers
        self.map_dendrograms = map_dendrograms
        self.display_weights = display_weights
        self.display_biases = display_biases
        self.training_step = training_step
        self.training_error = training_error
        self.validation_step = validation_step
        self.validation_error = validation_error
        self.layers = layers

    def grab_vars(self):
        grabvars = {
            'hinton': [],
            'dendro': [],
            'weights': [],
            'bias': []
        }

        # Hinton plot
        for layer in self.map_layers:
            if layer == 0:
                var = self.input
            else:
                var = self.layers[layer - 1].output
            grabvars['hinton'].append(var)

        # Dendrograms
        for layer in self.map_dendrograms:
            if layer == 0:
                var = self.input
            else:
                var = self.layers[layer - 1].output
            grabvars['dendro'].append(var)

        # Weights
        for layer in self.display_weights:
            grabvars['weights'].append(self.layers[layer - 1].weights)

        # Biases
        for layer in self.display_biases:
            grabvars['bias'].append(self.layers[layer - 1].biases)

        return grabvars

    def plot_hinton(self, map_layers, values):
        if len(values) > 0:
            for i, value in enumerate(values):  TFT.hinton_plot(np.array(value),
                                                                title="Hinton plot for " + str(map_layers[i]))

    def plot_dendrograms(self, map_dendrograms, values, targets):
        if len(values) > 0:
            for i, value in enumerate(values): TFT.dendrogram(value, targets,
                                                              title="Dendrogram for " + str(map_dendrograms[i]))

    def plot_weights(self, display_weights, values):
        if len(values) > 0:
            for i, value in enumerate(values): TFT.display_matrix(np.array(value),
                                                                  title="Weights for " + str(display_weights[i]))

    def plot_biases(self, display_biases, values):
        if len(values) > 0:
            for i, value in enumerate(values): TFT.display_matrix(np.array([value]),
                                                                  title="Biases for " + str(display_biases[i]))

    def visualize(self, only_error_plot=False):
        print('\n~~~~~~~~~~~~~~~~~~~~~~ Visualization ~~~~~~~~~~~~~~~~~~~~~')
        print('Drawing...')
        if only_error_plot:
            TFT.plot_train_val_error(self.training_step, self.training_error, self.validation_step, self.validation_error)
            print('Drawing completed!')
            plt.show()
            return

        TFT.plot_train_val_error(self.training_step, self.training_error, self.validation_step, self.validation_error)

        if self.map_batch_size is not 0:
            grabvars = self.grab_vars()
            inputs, targets = self.casemanager.get_minibatch(self.map_batch_size)
            matrix = self.session.run(grabvars, {self.input: inputs.tolist()})

            self.plot_hinton(self.map_layers, matrix['hinton'])
            self.plot_dendrograms(self.map_dendrograms, matrix['dendro'], targets)
            self.plot_weights(self.display_weights, matrix['weights'])
            self.plot_biases(self.display_biases, matrix['bias'])

        print('Drawing completed!')
        plt.show()

