import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from random import *

# Load MNIST data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Flatten and normalize the images
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Activation functions
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    x = np.clip(x, -500, 500)
    return x * (1.0 - x)

# Set activation function
activation_function = sigmoid

# Initialize weights for the neural network
input_size = 28 * 28
hidden_size = 28
output_size = 10

decimal_places = 3

#weights_input_hidden = np.random.rand(input_size, hidden_size)
#weights_hidden_output = np.random.rand(hidden_size, output_size)
weights_input_hidden = np.round(np.random.randn(input_size, hidden_size), decimal_places)
weights_hidden_output = np.round(np.random.randn(hidden_size, output_size), decimal_places)

# Training parameters
learning_rate = 0.1
epochs = 5

correct = 0
for i in range(len(test_images)):
    input_layer = test_images[i]

    hidden_layer_input = np.dot(input_layer, weights_input_hidden)
    hidden_layer_activation = activation_function(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
    output_layer_activation = activation_function(output_layer_input)

    predicted_label = np.argmax(output_layer_activation)
    true_label = np.argmax(test_labels[i])

    if predicted_label == true_label:
        correct += 1

accuracy = correct / len(test_images)
print("Accuracy on test data [Before training]: {:.2%}\n".format(accuracy))
#print(weights_input_hidden)
#print(weights_hidden_output)

# Training loop
for epoch in range(epochs):
    print("Beginning of epoch : " + str(epoch))
    for i in range(len(train_images)):
        # Forward pass
        input_layer = train_images[i]

        hidden_layer_input = np.dot(input_layer, weights_input_hidden)
        hidden_layer_activation = activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
        output_layer_activation = activation_function(output_layer_input)

        # Compute error
        output_error = train_labels[i] - output_layer_activation

        weights_hidden_output = np.round(weights_hidden_output, decimal_places)
        weights_input_hidden = np.round(weights_input_hidden, decimal_places)

        # Backpropagation
        weights_hidden_output += learning_rate * np.outer(hidden_layer_activation, output_error)
        weights_input_hidden += learning_rate * np.outer(input_layer, np.dot(output_error, weights_hidden_output.T) * hidden_layer_activation * (1 - hidden_layer_activation))
    
        weights_hidden_output = np.round(weights_hidden_output, decimal_places)
        weights_input_hidden = np.round(weights_input_hidden, decimal_places)

    # Testing the trained network
    correct = 0
    for i in range(len(test_images)):
        input_layer = test_images[i]

        hidden_layer_input = np.dot(input_layer, weights_input_hidden)
        hidden_layer_activation = activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_activation, weights_hidden_output)
        output_layer_activation = activation_function(output_layer_input)

        predicted_label = np.argmax(output_layer_activation)
        true_label = np.argmax(test_labels[i])

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / len(test_images)
    print("Accuracy on test data [epoch {}]: {:.2%}".format(epoch, accuracy))
    #print(weights_input_hidden)
    #print(weights_hidden_output)
