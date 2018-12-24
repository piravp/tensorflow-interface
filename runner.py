from src.ANN import ANN
from src.Caseman import Caseman
from src.presets import *

def runner(cfunc, dimensions, hidden_activation, output_activation, cost_func,
                 lrate, initial_weight_range, optimizer, case_fraction, val_frac, val_interval, test_frac, mb_size,
                 map_batch_size, steps, map_layers, map_dendro, display_weights, display_biases):

    casemanager = Caseman(cfunc=cfunc, case_fraction=case_fraction, vfrac=val_frac, tfrac=test_frac)

    ann = ANN(cman=casemanager, dimensions=dimensions, hidden_activation=hidden_activation, output_activation=output_activation,
               cost_func=cost_func, lrate=lrate, initial_weight_range=initial_weight_range, optimizer=optimizer, mb_size=mb_size,
               steps=steps, val_interval=val_interval, map_batch_size=map_batch_size, map_layers=map_layers, map_dendro=map_dendro,
               display_weights=display_weights, display_biases=display_biases)

    ann.run()


runner(**mnist)


# Note
# * Normalization range should depend on choice of activation function. https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/30