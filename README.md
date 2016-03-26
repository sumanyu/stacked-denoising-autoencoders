# stacked-autoencoders

Contains [scikit-learn](http://scikit-learn.org/stable/) wrappers for `StackedDenoisingAutoencoders` and `DenoisingAutoencoder`.

`StackedDenoisingAutoencoders` are trained greedily layer by layer. 

See example usage by running `python run_mnist_ae.py` which trains a single layer autoencoder on the MNIST dataset and uses the latent features as input to a naive bayes classifier.

Similarly, run `python run_mnist_stacked_ae.py` for a stacked two layer autoencoder.

Tested with:
* `Python 2.7.9`
* `Theano 0.7.0`
* `Scikitlearn 0.17`
* `Numpy 1.10.2`
