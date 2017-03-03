import numpy as np


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

        self.learning_rate = learning_rate

        self.input_layer_output_weights = \
            (np.random.rand(self.hnodes, self.inodes)-0.5)
        self.hidden_layer_output_weights = \
            (np.random.rand(self.onodes, self.hnodes)-0.5)
        pass

    def train(self):
        pass

    def query(self):
        pass