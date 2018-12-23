import tensorflow as tf
import sklearn.preprocessing as sklearnp
from src.helpers.tflowtools import show_image
import src.helpers.tflowtools as TFT
import numpy as np

##########################################################################################
                                # ***** Data functions ******
##########################################################################################


mnist = lambda : extract(preprocess(tf.keras.datasets.mnist.load_data(), 'mnist'), 'mnist')
autoencoder = lambda : extract(preprocess(TFT.gen_dense_autoencoder_cases(count=10000, size=8, dr=(0, 1)), 'autoencoder'), 'autoencoder')



#--------------- Helper functions ---------------#
def preprocess(data, type):
    scaler = sklearnp.MinMaxScaler()

    if type=='mnist':
        # Note that fitting (finding min/max for each column) is only done for training data.
        # Transforming (applying the min/max=dividing each entry by max) is done on both
        # train and test set. Fitting both individually would be considered cheating as
        # we're pretending we don't know how the test data will look like.
        (train_x, train_y), (test_x, test_y) = data

        data_x = np.append(train_x, test_x, axis=0)
        data_y = np.append(train_y, test_y, axis=0)

        # Fit and transform training data
        samples_train, pixels, _ = data_x.shape                         # 60000, 28, 28     (3d array)
        data_x = data_x.reshape(samples_train, pixels*pixels)           # reshape to 60000, 28*28 as scaler expects 2d array
        data_x = scaler.fit_transform(data_x.astype(float))

        dataset = []
        for i in range(len(data_x)):
            item = data_x[i], data_y[i]
            dataset.append(item)


        # Normalized dataset
        return dataset

    if type=='autoencoder':
        processed_dataset = []
        for input_target in data:
            x, y = input_target
            x, y = np.asarray(x), np.asarray(y)
            processed_dataset.append((x, y))

        return processed_dataset



# Extract info on n_classes and n_features
def extract(dataset, type):
    if type == 'mnist':
        # Classes
        targets = []
        for _, target in dataset:
            targets.append(target)
        n_targets = len(set(targets))
        # Features
        n_features = len(dataset[0][0])

        return dataset, n_features, n_targets

    if type == 'autoencoder':
        n_features = n_targets = len(dataset[0][0])

        return dataset, n_features, n_targets


# temp = mnist
# (train_x, train_y), (test_x, test_y) = mnist()
# show_image(train_x[0])

# m1, m2, m3 = mnist()
# a1, a2, a3 = autoencoder()
# print(m2, m3)
# print(a2, a3)


