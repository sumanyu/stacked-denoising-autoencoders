import os

from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

from DenoisingAutoencoder import DenoisingAutoencoder
from StackedDenoisingAutoencoders import StackedDenoisingAutoencoders


custom_data_home = os.path.join(os.path.split(__file__)[0], "data")
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)


X, y = mnist.data / 255., mnist.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Stacked AE test

stacked_ae = StackedDenoisingAutoencoders(hidden_layers_sizes=[400, 200], verbose=True, training_epochs=5)
stacked_ae.fit(X_train)

X_train_latent = stacked_ae.transform_latent_representation(X_train)
X_test_latent = stacked_ae.transform_latent_representation(X_test)

clf = MultinomialNB()

# Fit the model
clf.fit(X_train_latent, y_train)

# Perform the predictions
y_predicted = clf.predict(X_test_latent)


from sklearn.metrics import accuracy_score
print "Accuracy = {} %".format(accuracy_score(y_test, y_predicted)*100)

from sklearn.metrics import classification_report
print "Classification Report \n {}".format(classification_report(y_test, y_predicted, labels=range(0,10)))
