__author__ = 'vps'

import timeit

from sklearn.base import BaseEstimator
from DenoisingAutoencoder import DenoisingAutoencoder


class StackedDenoisingAutoencoders(BaseEstimator):
    def __init__(self, hidden_layers_sizes, learning_rate=0.1, training_epochs=15, corruption_level=0.0, batch_size=20, verbose=False):
        self.n_visible = None
        self.hidden_layers_sizes = hidden_layers_sizes
        self.learning_rate = learning_rate
        self.corruption_level = corruption_level
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.das = []

    def fit(self,X):
        start_time = timeit.default_timer()
        self.n_visible = X.shape[1]

        for hidden_layer_size in self.hidden_layers_sizes:
            da = DenoisingAutoencoder(hidden_layer_size,
                                      self.learning_rate,
                                      self.training_epochs,
                                      self.corruption_level,
                                      self.batch_size,
                                      self.verbose)
            self.das.append(da)

        # Greedily train the AE
        input = X
        for da in self.das:
            da.fit(input)
            input = da.transform_latent_representation(input)

        end_time = timeit.default_timer()
        training_time = (end_time - start_time)

        print(("The %d%% corruption code " % (self.corruption_level*100) +
               ' ran for %.2fm' % (training_time / 60.)))

    def transform(self, X):
        input = X
        for da in self.das:
            input = da.transform(input)

        return input

    def transform_latent_representation(self, X):
        input = X
        for da in self.das:
            input = da.transform_latent_representation(input)

        return input
