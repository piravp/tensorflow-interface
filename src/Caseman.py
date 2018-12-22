import numpy as np
import src.functions.data as data

# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system



class Caseman():

    def __init__(self, cfunc, case_fraction=1.0, vfrac=0.0, tfrac=0.2):
        self.casefunc = cfunc                                       # Function for generating cases
        self.case_fraction = case_fraction
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)

        # Run functions
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        # Run case generator
        self.cases = self.casefunc()                                # case = (input-vector, target-vector)

    def organize_cases(self):
        # cases = np.array(self.cases)
        cases = self.cases
        # Whole dataset?
        n_cases = round(len(cases) * self.case_fraction)
        cases = cases[0:n_cases]

        # Randomly shuffle all cases
        np.random.shuffle(cases)

        # Separators
        training_end = round(len(cases) * self.training_fraction)
        validation_end = training_end + round(len(cases)*self.validation_fraction)

        # Split into train-, test- and validation sets
        self.training_cases = cases[0:training_end]
        self.validation_cases = cases[training_end:validation_end]
        self.testing_cases = cases[validation_end:]

    # Returns a random minibatch
    def get_minibatch(self, size):
        np.random.shuffle(self.training_cases)
        minibatch = self.training_cases[:size]
        return minibatch

    def get_training_cases(self): return self.training_cases
    def get_validation_cases(self): return self.validation_cases
    def get_testing_cases(self): return self.testing_cases


cm_mnist = Caseman(cfunc=data.mnist, case_fraction=1.0, vfrac=0.0, tfrac=0.2)
cm_auto = Caseman(cfunc=data.autoencoder, case_fraction=1.0, vfrac=0.0, tfrac=0.2)

for case in cm_mnist.get_training_cases():
    x, y = case

n_training = len(cm_mnist.training_cases)
print(n_training)
# mb = cm_mnist.get_minibatch(10)
# print(len(mb))