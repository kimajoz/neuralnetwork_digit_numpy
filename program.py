import neuralnetwork as nn
import numpy as np

layer_sizes = (2,3,5,2)
#layer_sizes = (1000,500,10)
x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)
