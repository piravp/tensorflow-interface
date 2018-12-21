import tensorflow as tf
import sklearn.preprocessing as sklearnp
from src.helpers.tflowtools import show_image

m = tf.keras.datasets.mnist

##########################################################################################
                                # ***** Data functions ******
##########################################################################################


mnist = lambda : normalize(m.load_data())



#--------------- Helper functions ---------------#

def normalize(data, mnist=True):
    scaler = sklearnp.MinMaxScaler()

    if mnist:
        # Note that fitting (finding min/max for each column) is only done for training data.
        # Transforming (applying the min/max by diving each entry by max) is done on both
        # train and test set. Fitting both individually would be considered cheating as
        # we're pretending we don't know how the test data is going to look like.
        (train_x, train_y), (test_x, test_y) = data

        # Fit and transform training data
        samples_train, pixels, _ = train_x.shape                         # 60000, 28, 28     (3d array)
        train_x = train_x.reshape(samples_train, pixels*pixels)          # reshape to 60000, 28*28 as scaler expects 2d array
        normalized_train = scaler.fit_transform(train_x), train_y

        # Transform test data
        samples_test, _, _ = test_x.shape
        test_x = test_x.reshape(samples_test, pixels*pixels)
        normalized_test = scaler.transform(test_x), test_y

        # Normalized train- and test set
        normalized_data = normalized_train, normalized_test
        return normalized_data


    normalized = scaler.fit_transform(data)
    return normalized



temp = mnist
(train_x, train_y), (test_x, test_y) = mnist()
show_image(train_x[0])