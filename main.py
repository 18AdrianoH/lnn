import neuralNet as nn

# create a new neural net
input_nodes = 3
output_nodes = 3
hidden_nodes = 3

learning_rate = 0.3

net = nn.NeuralNet(input_nodes, output_nodes, hidden_nodes, learning_rate);