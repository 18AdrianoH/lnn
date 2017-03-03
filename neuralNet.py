import numpy as np
import scipy.special

# shitty simple neural network class
# it has one input layer, one output layer, and a single hidden layer
# nodes are connected to all subsequent nodes where such is possible


class NeuralNet:
    # constructor
    # each parameter is a number representing the number of given objects
    def __init__(self, input_nodes, output_nodes, hidden_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.lr = learning_rate

        # set the weights to random values within a gaussian distribution
        # this way we get different weights, but none which bias or saturate the system
        self.wih = \
            np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.who = \
            np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        self.activation_function = lambda x: \
            scipy.special.expit(x)  # activation is a sigmoid
        pass

    # one iteration of training given inputs and desired targets
    def train(self, inputs_list, targets_list):
        # inputs and targets
        targets = np.array(targets_list, ndmin=2).T
        inputs = np.array(inputs_list, ndmin=2).T
        # outputs
        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))
        outputs = self.activation_function(np.dot(self.who, hidden_outputs))

        # error
        output_errors = (targets-outputs)

        # back-propagated hidden layer error
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += \
            self.lr * np.dot((output_errors * outputs * (1.0 - outputs)), np.transpose(hidden_outputs))
        self.wih += \
            self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), (np.transpose(inputs)))
        pass

    # function that will query the neural net with an input list,
    # returning the outputs for the given inputs
    def query(self, inputs_list):
        # we convert our list of inputs into a matrix
        inputs = np.array(inputs_list, ndmin=2).T

        # outputs of hidden layer are as follows:
        # dot product of input matrix by first layer weights matrix
        # passed through our sigmoid activation lambda function
        hidden_outputs = self.activation_function(np.dot(self.wih, inputs))

        # final outputs' code is more or less the same as hidden outputs'
        final_outputs = self.activation_function(np.dot(self.who, hidden_outputs))
        return final_outputs
