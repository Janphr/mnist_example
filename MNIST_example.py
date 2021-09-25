from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import os
import numpy as np
from _framework import *


##################################################################################################
# This class's purpose only is to load the mnist dataset and prepare the data for the training
class MnistLoader:
    cwd = os.getcwd()

    def __init__(self, scaling, t_clipping):
        self.x_train, self.t_train = loadlocal_mnist(
            images_path=self.cwd + '/mnist/train-images.idx3-ubyte',
            labels_path=self.cwd + '/mnist/train-labels.idx1-ubyte')

        self.x_test, self.t_test = loadlocal_mnist(
            images_path=self.cwd + '/mnist/t10k-images.idx3-ubyte',
            labels_path=self.cwd + '/mnist/t10k-labels.idx1-ubyte')

        self.scaling = scaling
        self.t_clipping = t_clipping

    def get_train_data(self):
        t_train = np.array(self.t_train)

        # transform labels into one hot representation
        targets_one_hot = np.identity(t_train.max() + 1, dtype=float)[t_train]

        if self.t_clipping:
            # we don't want zeroes and ones in the labels neither:
            targets_one_hot[targets_one_hot == 0] = 0.01
            targets_one_hot[targets_one_hot == 1] = 0.99

        return np.asfarray(self.x_train) * self.scaling + 0.01, targets_one_hot

    def get_test_data(self):
        t_test = np.array(self.t_test)

        # transform labels into one hot representation
        targets_one_hot = np.identity(t_test.max() + 1, dtype=float)[t_test]

        if self.t_clipping:
            # we don't want zeroes and ones in the labels neither:
            targets_one_hot[targets_one_hot == 0] = 0.01
            targets_one_hot[targets_one_hot == 1] = 0.99

        return np.asfarray(self.x_test) * self.scaling + 0.01, targets_one_hot


##################################################################################################
# The following shows, how to use our Framework

# Configure MNIST Loader, so that values are scaled to 0-1 and clipped to 0.01 and 0.99
l = MnistLoader(scaling=0.99 / 255, t_clipping=True)

# extract train and test data (-> already scaled)
x_train, t_train = l.get_train_data()
x_test, t_test = l.get_test_data()

# create and configure the ANN
layer_arr = []

layer_arr.append(FullyConnected(x_test.shape[1], 100))  # hidden layer -> size 100
layer_arr.append(Tanh())  # Activation function has to be added as layer
# layer_arr.append(Dropout(100, 0.7)) # optionally Dropout can be added by a corresponding layer

layer_arr.append(FullyConnected(100, 50))  # hidden layer -> size 50
layer_arr.append(Tanh())  # corresponding activation function

layer_arr.append(FullyConnected(50, t_test.shape[1]))  # output layer -> size depending on targets
layer_arr.append(Tanh())  # corresponding activation function

# create network
network = Network(*layer_arr)

# create trainer
learning_rate = 0.0001
alpha = 0.001
trainer = Trainer(network, cce_loss, AdamOptimizer(learning_rate, alpha))

# train network
epochs = 10
batch_size = 1000

print("Starting Training with batch size " + str(batch_size) + " and " + str(epochs) + " epochs ...")
loss, validation_loss = trainer.train(x_train, t_train, epochs, batch_size, live_eval=(x_test, t_test))
print("Training done!")

# test network
# note that test and train always apply softmax 
avg_tps, avg_confidence, correct_wrong_predictions, confusion_matrix = network.test(x_test, t_test)

# print results
print("Test done!")
print("Accuracy: " + str(avg_tps))
print("Confidence: " + str(avg_confidence))

# for further metrics e.g. F1 score use confusion_matrix

# configure matplotlib
matplotlib.use("TkAgg")
params = {"ytick.color": "k",
          "xtick.color": "k",
          "axes.labelcolor": "k",
          "axes.edgecolor": "k",
          "text.color": 'k'}
plt.rcParams.update(params)

# plot learning curve
plt.plot(loss, label="loss")
plt.plot(validation_loss, label="validation loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
# ax = plt.gca()
# ax.set_facecolor('xkcd:salmon')
plt.show()
