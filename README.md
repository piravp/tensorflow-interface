# TensorFlow Interface
An interface to TensorFlow allowing to build general artificial neural networks. Tweakable hyperparameters include number of hidden layers, neurons in each layer, hidden activation function, cost function, optimizer function etc. The full list can be found in the table below.

# Usage
### Install
The easiest is to have Anaconda with Python 3.6, numpy, scipy and Tensorflow installed

### Run
The interface can be accessed through `runner.py`. Considering there are many tweakable parameters, each unique network is added to `presets.py` which can be directly passed to `runner.py` like this:

```python
from presets import *
# ...
runner(**mnist)
```


### Parameters
|       **Name**                |   **Type**     |                  **Description**                                                                                                                                                                               |
|:-----------------------------:|:--------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|       *cfunc*                 |  `function`    | Reference to a lambda function that generates the dataset when called in the casemanager.|
|       *dimensions*            |  `array<int>`  | The number of layers in the network along with the size of each layer|
|       *hidden_activation*     |  `function`    | Function to be used for all hidden layers (i.e., all layers except the input and output).|
|       *output_activation*     |  `function`    | This is often different from the hidden-layer activation function; for example, it is common to use softmax for classification problems, but only in the final layer.|
|       *cost_func*             |  `function`    | (a.k.a. loss function) Defines the quantity to be minimized, such as mean-squared error or cross-entropy.|
|       *optimizer*             |  `function`    | Tie together the loss function and model parameters by updating the model in response to the output of the loss function.|
|       *lrate*                 |  `float`       | The same rate can be applied to each set of weights and biases throughout the network and throughout the training phase. More complex schemes, for example those that reduce the learning rate throughout training, are possible but not required.|
|       *initial_weight_range*  |  `array<float>`| Two real numbers, an upper and lower bound, to be used when randomly initializing all weights (including those from bias nodes) in the network. *Optionally, this parameter may take a value such as the word scaled, indicating that the weight range will be calculated dynamically, based on the total number of neurons in the upstream layer of each weight matrix. (This feature is not implemented yet)*|
|       *case_fraction*         |  `float`       | Some data sources (such as MNIST) are very large, so it makes sense to only use a fraction of the total set for the combination of training, validation and testing. This should default to 1.0, but much lower values can come in handy for huge data files.|
|       *vaL_frac*              |  `float`       | The fraction of data cases to be used for validation testing|
|       *val_interval*          |  `int`         | Number of training minibatches between each validation test.|
|       *test_frac*             |  `float`       | The fraction of the data cases to be used for standard testing (i.e. after training has finished).|
|       *mb_size*               |  `int`         | The number of training cases in a minibatch|
|       *match_batch_size*      |  `int`         | The number of training cases to be used for a map test (described below). A value of zero indicates that no map test will be performed.|
|       *steps*                 |  `int`         | The total number of minibatches to be run through the system during training.|
|       *map_layers*            |  `array<int>`  | The layers to be visualized during the map test.|
|       *map_dendro*            |  `array<int>`  | List of the layers whose activation patterns (during the map test) will be used to produce dendrograms, one per specified layer.|
|       *display_weights*       |  `array<int>`  | List of the weight arrays to be visualized at the end of the run.|
|       *display_biases*        |  `array<int>`  | List of the bias vectors to be visualized at the end of the run.|

## Visualization
### Mapping
This involves taking a small sample of data cases (e.g. 10-20 examples) and running them
through the network, with learning turned off. The activation levels of a user-chosen set of layers are
then displayed for each case.

### Dendrograms
For any given network layer, a comparison of the different activation vectors (across all
cases of a mapping) can then serve as the basis for a dendrogram, a convenient graphic indicator of
the network’s general ability to partition data into relevant groups.


### Weight and bias viewing
Simple graphic views of the weights and/or biases associated with the connections between any user-chosen pairs of layers.