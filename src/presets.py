import src.functions.data as data
import src.functions.cost as cost
import src.functions.activation as activation
import src.functions.optimizer as optimizer

mnist = {
    'cfunc': data.mnist,
    'dimensions': [170],
    'hidden_activation':  activation.relu,
    'output_activation': activation.softmax,
    'cost_func': cost.mse,
    'lrate': 0.001,
    'initial_weight_range': [-0.1, 0.1],
    'optimizer': optimizer.adam,
    'case_fraction': 1.0,
    'val_frac': 0.1,
    'val_interval': 100,
    'test_frac': 0.1,
    'mb_size': 100,
    'map_batch_size': 0,
    'steps': 10000,
    'map_layers': [],
    'map_dendro': [],
    'display_weights': [],
    'display_biases':  []
}

autoencoder = {
    'cfunc': data.autoencoder,
    'dimensions': [100],
    'hidden_activation':  activation.relu,
    'output_activation': activation.softmax,
    'cost_func': cost.mse,
    'lrate': 0.001,
    'initial_weight_range': [-0.1, 0.1],
    'optimizer': optimizer.adam,
    'case_fraction': 1.0,
    'val_frac': 0.1,
    'val_interval': 100,
    'test_frac': 0.1,
    'mb_size': 100,
    'map_batch_size': 0,
    'steps': 10000,
    'map_layers': [],
    'map_dendro': [],
    'display_weights': [],
    'display_biases':  []
}