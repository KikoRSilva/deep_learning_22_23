#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

# Auxiliary functions

def one_hot_vector(y, n_classes):
    one_hot = np.zeros(n_classes)
    one_hot[y] = 1
    return one_hot.T

def softmax(x):
    f = x - np.max(x)
    return np.exp(f) / np.sum(np.exp(f))

def relu(x):
    return np.maximum(0, x)

def reluDerivative(x):
    return (x > 0) * 1

# Models

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        self.mistakes = 0
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):

    def signum(self, net):
        if net >= 0:
            return 1
        return -1


    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        eta = 1
        
        y_hat = self.predict(x_i)

        if y_hat != y_i:
            self.mistakes += 1
            self.W[y_i, :] += eta * x_i
            self.W[y_hat, :] -= eta * x_i


class LogisticRegression(LinearModel):

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """

        class_probs = softmax(np.dot(self.W, x_i.T)[:, None])
        y_one_hot = one_hot_vector(y_i, self.W.shape[0])[:, None]

        grad_i = (y_one_hot - class_probs) * x_i[None, :]

        self.W += learning_rate * grad_i



class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_hidden_layers):
        # Initialize an MLP with a single hidden layer.
        struct_size = [n_features] + [hidden_size for _ in range(n_hidden_layers)] + [n_classes]

        self.n_layers = len(struct_size) - 1
        self.weights = []
        self.biases = []
        for l in range(self.n_layers):
            self.weights.append(np.random.normal(loc=0.1, scale=0.1, size=(struct_size[l+1], struct_size[l])))
            self.biases.append(np.zeros(struct_size[l+1]))

        self.g = relu                       # activation function for hidden layers
        self.d_g = reluDerivative           # derivate for g
        self.o = softmax                    # activation function for output layer

        self.loss = lambda y, prd_probs: -y.dot(np.log(prd_probs))  # cross-entropy loss fuction
        self.grad_loss = lambda y, prd_probs: prd_probs - y         # gradient of this loss function


    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        prd_labels = []
        for x_i in X:
            h = x_i
            for l in range(self.n_layers - 1):
                z = np.dot(self.weights[l], h) + self.biases[l]
                h = self.g(z)
            z_output = np.dot(self.weights[-1], h) + self.biases[-1]
            prd_probs = self.o(z_output)
            prd_labels.append(prd_probs)
        prd_labels = np.array(prd_labels).T
        prd_labels = prd_labels.argmax(axis=0)
        
        return prd_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):

        for x_i, y_i in zip(X, y):
            # FEED-FORWARD PROPAGATION
            hidden_values = []  # z and h pairs
            for l in range(self.n_layers - 1):
                z = np.dot(self.weights[l], hidden_values[l][1] if l > 0 else x_i) + self.biases[l]
                hidden_values.append((z, self.g(z)))

            z_output = np.dot(self.weights[-1], hidden_values[-1][1]) + self.biases[-1]
            
            prd_probs = self.o(z_output)
            y_one_hot = one_hot_vector(y_i, z_output.shape[0])

            # error = self.loss(y_i, prd_probs)

            # BACKWARD PROPAGATION
            grad_z = self.grad_loss(y_one_hot, prd_probs)

            for l in range(self.n_layers - 1, -1, -1):
                h = x_i if l == 0 else hidden_values[l-1][1]

                grad_w = np.dot(grad_z[:, None], h[:, None].T)
                grad_b = grad_z

                if l > 0:
                    grad_h = np.dot(self.weights[l].T, grad_z)
                    grad_z = grad_h * self.d_g(hidden_values[l-1][0])


                self.weights[l] -= learning_rate * grad_w
                self.biases[l] -= learning_rate * grad_b

            

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
