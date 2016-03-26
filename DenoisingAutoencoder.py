from __future__ import print_function

from sklearn.base import BaseEstimator
from dA import *
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import timeit
import os
import sys


class DenoisingAutoencoder(BaseEstimator):
    def __init__(self, n_hidden, learning_rate=0.1, training_epochs=15, corruption_level=0.0, batch_size=20, verbose=False):
        self.n_visible = None
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.corruption_level = corruption_level
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.da = None
        self.x = T.matrix('x')

    def load_data(self,X):
        try:
            matrix = X.as_matrix()
        except AttributeError:
            matrix = X
        shared_x = theano.shared(numpy.asarray(matrix, dtype=theano.config.floatX), borrow=True)
        return shared_x

    def fit(self,X):
        start_time = timeit.default_timer()
        self.n_visible = X.shape[1]

        train_set_x = self.load_data(X)

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        self.da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=self.x,
            n_visible=self.n_visible,
            n_hidden=self.n_hidden
        )

        cost, updates = self.da.get_cost_updates(
            corruption_level=self.corruption_level,
            learning_rate=self.learning_rate
        )

        train_da = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        # training...
        for epoch in xrange(self.training_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))

            if self.verbose:
                print('Training epoch %d, cost ' % epoch, numpy.mean(c))

        end_time = timeit.default_timer()
        training_time = (end_time - start_time)

        print(("The %d%% corruption code " % (self.corruption_level*100) +
               ' ran for %.2fm' % (training_time / 60.)), file=sys.stderr)

    def transform(self, X):
        z = self.da.get_prediction()
        predict_da = theano.function([self.x], z)
        return predict_da(X)

    def transform_latent_representation(self, X):
        h = self.da.get_latent_representation()
        predict_da = theano.function([self.x],h)
        return predict_da(X)

