import network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Define the network construct
net = network.Network([784, 30, 10])

# Stochastic gradient descent
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
