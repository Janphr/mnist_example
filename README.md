This repository is an example of how to wrap the submodule _framework in a minimal project. Following the content of the README.md of the submodule:

# Machine Learning Framework Package

This repository contains the Machine Learning Framework Package. It comprises different layer types, like the standard fully connected layer, activation layers (relu, sigmoid and tanh) and the dropout layer, as well as multiple optimizers, like Adam, Adagrad, Stochastic Gradient Descent, Momentum based Gradient Descent and Nesterov's accelerated Gradient Descent. All paths mentioned in this document are relative to the project directory.

## How to deploy our code
1. clone directory via https with
`git clone https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/lectures/machine-learning/student-material/ws20/Team-08/framework.git` 


2. execute `git submodule update --remote --merge` to fetch the latest changes from upstream submodule _framework, merge them in, and check out the latest revision of the submodule.

3. Install python

4. Create an virtual environment of your choice. We use pipenv. Tutorials can be found in the web.

5. Install the required packages with `pip install -r requirements.txt`

## Creating a Network

The  `class Network(Layer)` is implemented in */_framework/network.py*. An instance is easily created by calling the constructer with a list of layers as parameter:

`network = Network(*[FullyConnected(100,50), Tanh(), FullyConnected(50,10), Tanh()])`

A network supports forward an backward propagation, as well as a simple prediction function `predict(self, x)` (basically forward propagation with softmax) and a function `test(self, x, t)`, which tests the network for an input `x` and targets `t` and determines the accuracy, the confidence and the confusion matrix.

## Creating the Layers

There are different types of layers, which the network can consist of. Each layer provides forward / backward propagation and an update function for the optimizer. If instances of `Network` and `Trainer` are used, these functions don't have to be used outside the framework.

The most important layer is the `class FullyConnected(Layer)` (*/_framework/layers/fullyConnected.py*). It can be created by calling the constructor with the input and output sizes as parameters:

`fc_layer = FullyConnected(100,50)`

The activation layers can be created, simply by calling the constructor (e.g. `Tanh()`, `Sigmoid()` or `ReLU()`)

The dropout layer needs a size and a propability *p* as an argument. `dropout_layer = Dropout(100, 0.7)` for example creates a layer of size 100, where only 70% of the connections to the next layer are remaining.

The corresponding source files are stored in */_framework/layers/*. Note that softmax is no layer in this framework, but is automatically applied as a function when using the `network.predict(x)` function or the trainer.

## Creating a Trainer to Train a Network

### Initialization

The `class Trainer` (*/_framework/trainer.py*) provides the functionality to train a network. Therefor, the trainer has to be given a network, a loss function and an optimizer. An initialization could look like the following:

`trainer = Trainer(network, cce_loss, AdamOptimizer(learning_rate, alpha))`

This framework provides binary cross entropy, categorical cross entropy and mean squared error as loss functions.
The following optimizers are available:
* `AdamOptimizer`: parameters are *learning rate, alpha, gamma, beta1* and *beta2*.
* `AdagradOptimizer`: parameters are *learning rate, alpha* and *gamma*.
* `NAGOptimizer` (Nesterov Accelerated Gradient): parameters are *learning rate, alpha* and *gamma*.
* `SGDOptimizer`(Stochastic Gradient Descent): parameters are *learning rate* and *alpha*.
* `MGDOptimizer` (Momentum based Gradient Descent): parameters are *learning rate, alpha* and *gamma*.

Regularization is supported by all optimizers via the parameter *alpha*.

### Training

In order to train a network, the function `trainer.train(x, target, epochs, batch_size, live_eval=None)` has to be used. 
* `x`: input/feature matrix
* `target`: target matrix (one hot vectors for multiclass classification)
* `epochs`: number of epochs to train
* `batch_size`: number of samples per batch
* `live_eval`: tuple of input and target matrix of a validation dataset for live evaluation

The function returns a tuple of the losses of the training set and the validation set (if there is one specified) after each epoch as lists. This output could be used for visualization of the learning curve.
An example could look like this:
```python
epochs = 100
batch_size = 1000
loss, validation_loss = trainer.train(x_train, t_train, epochs, batch_size, live_eval=(x_test, t_test))
```

## Using Utils

### */_framework/utils/object_IO.py*
* `save_network(network, name)` saves a network to the */data/trained_networks* folder
* `delete_network(name)` deletes a network that is saved to */data/trained_networks*
* `load_network(name)` loads a network that is saved to */data/trained_networks*

### */_framework/utils/evaluations.py*
This file contains multiple functions and classes that are used to connect the framework to a web based frontend. This comprises classes to plot the learning curve, the confusion matrix and input features.

## Example
An example of how to use our code is provided in *MNIST_example.py*. In this script, firstly a class is defined that helps to load the MNIST data and preprocess it, so that it can be directly fed into neural nets.
Below, a demonstration of the functions of this framework can be found.

A neural net is created with 784 x 100 x 50 x 10 Neurons. The activation function is *tanh*.
Then, a trainer is created with an `AdamOptimizer` and categorical cross entropy as loss function. The regularization parameter `alpha` and the learning rate are set manually, while the default values are used for the other hyperparameters.

Afterwards, the training is perfomed on the test dataset with a batch size of 1000 over 10 epochs. The validation datasets are taken for live evaluation.

When the training is finished, the network ist testet on the validation dataset. The resulting accuracy and confidence are printed to the console. The learning curve is visualized using *matplotlib*. Further evaluations could be performed using the `confusion_matrix`.
