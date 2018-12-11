import neuralnetwork as nn
import numpy as np
import matplotlib.pyplot as plt

with np.load('mnist.npz') as data:
    #print(data.files)
    training_images = data['training_images']
    training_labels = data['training_labels']

plt.imshow(training_images[0].reshape(28,28), cmap = 'gray')
plt.show()
print(training_labels[0])

#layer_sizes = (2,3,5,2) #basic
#layer_sizes = (1000,500,10) #big
layer_sizes = (784,5,10) #reals

x = np.ones((layer_sizes[0],1))

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(x)
