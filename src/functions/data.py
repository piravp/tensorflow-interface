import tensorflow as tf
import sklearn.preprocessing as sklearnp
from src.helpers.tflowtools import show_image
import src.helpers.tflowtools as TFT
import numpy as np

##########################################################################################
                                # ***** Data functions ******
##########################################################################################

#TODO: Split normalization to its own function

mnist = lambda : preprocess_wrapper(tf.keras.datasets.mnist.load_data(), 'mnist')
autoencoder = lambda : preprocess_wrapper(TFT.gen_dense_autoencoder_cases(count=10000, size=8, dr=(0, 1)), 'autoencoder')



#--------------- Helper functions ---------------#
def preprocess(dataset, type):
    scaler = sklearnp.MinMaxScaler()

    if type=='mnist':
        # Note that fitting (finding min/max for each column) is only done for training data.
        # Transforming (applying the min/max=dividing each entry by max) is done on both
        # train and test set. Fitting both individually would be considered cheating as
        # we're pretending we don't know how the test data will look like.
        (train_x, train_y), (test_x, test_y) = dataset

        data_x = np.append(train_x, test_x, axis=0)
        data_y = np.append(train_y, test_y, axis=0)

        # Fit and transform training data
        samples_train, pixels, _ = data_x.shape                         # 60000, 28, 28     (3d array)
        data_x = data_x.reshape(samples_train, pixels*pixels)           # reshape to 60000, 28*28 as scaler expects 2d array
        data_x = scaler.fit_transform(data_x.astype(float))

        dataset = []
        # Gå fra to separate lister (input:0, target:1) til én liste med tuples
        for i in range(len(data_x)):
            # TODO: Done 24.12.2018, 00:34 --> convert ndarray to list
            # item = data_x[i].tolist(), data_y[i]

            target_as_one_hot = TFT.int_to_one_hot(int=data_y[i], size=10)
            item = data_x[i], target_as_one_hot
            dataset.append(item)


        # Normalized dataset
        return dataset

    if type=='autoencoder':
        processed_dataset = []
        for input_target in dataset:
            x, y = input_target
            # np.asarray(input_target).T.tolist()   # reshape (2, 8) to (8, 2)
            # x, y = np.asarray(x), np.asarray(y)
            processed_dataset.append((x, y))

        return processed_dataset



# Extract info on n_classes and n_features
def extract(dataset, type):
    if type == 'mnist':
        # Classes
        targets = []
        for _, target in dataset:
            targets.append(target)
        # n_targets = len(set(targets))
        n_targets = len(targets[0])
        # Features
        n_features = len(dataset[0][0])

        return dataset, n_features, n_targets

    if type == 'autoencoder':
        n_features = n_targets = len(dataset[0][0])

        return dataset, n_features, n_targets


def preprocess_wrapper(data, type):
    if type == 'mnist':
        preprocessed = preprocess(dataset=data, type='mnist')
        extracted = extract(dataset=preprocessed, type='mnist')
        return extracted

    if type == 'autoencoder':
        preprocessed = preprocess(dataset=data, type='autoencoder')
        extracted = extract(dataset=preprocessed, type='autoencoder')
        return extracted


# temp = mnist
# (train_x, train_y), (test_x, test_y) = mnist()
# show_image(train_x[0])

# m1, m2, m3 = mnist()
# a1, a2, a3 = autoencoder()
# print(m2, m3)
# print(a2, a3)


