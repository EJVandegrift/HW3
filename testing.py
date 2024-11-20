import numpy as np
import random
import math
import graph


def MCCE_Loss(true_label, predictions, num_dif_labels=2):
    '''
    MCCE Loss for a single observation
    '''
    if hasattr(predictions, 'tolist'):
        predictions = predictions.flatten().tolist()
    
    # Clip predictions to prevent log(0)
    predictions = [max(min(p, 1-1e-15), 1e-15) for p in predictions]
    
    hot_encoded_labels = [1 if true_label == i else 0 for i in range(num_dif_labels)]
    total = 0

    for i in range(num_dif_labels):
        total += hot_encoded_labels[i] * math.log(predictions[i])

    return total * -1

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0, keepdims=True)

class Manual_nn():
    def __init__(self, k, epoch, learning_rate, num_hid_layers=1):
        self.k = k
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.num_hid_layers = num_hid_layers
        self.node_layers = []
        self.weights_arrays = []
        self.bias_arrays = []

    def init_weights_and_bias(self, num_features=2, num_outputs=2):
        self.weights_arrays = []
        self.bias_arrays = []
        
        # Xavier/Glorot initialization
        for i in range(self.num_hid_layers + 1):
            if i == 0:
                # First layer weights
                scale = np.sqrt(2.0 / (num_features + self.k))
                new_weights = np.random.normal(0, scale, size=(self.k, num_features))
                new_bias = np.zeros((self.k, 1))
                
            elif i == self.num_hid_layers:
                # Output layer weights
                scale = np.sqrt(2.0 / (self.k + num_outputs))
                new_weights = np.random.normal(0, scale, size=(num_outputs, self.k))
                new_bias = np.zeros((num_outputs, 1))
                
            else:
                # Hidden layer weights
                scale = np.sqrt(2.0 / (self.k + self.k))
                new_weights = np.random.normal(0, scale, size=(self.k, self.k))
                new_bias = np.zeros((self.k, 1))
                
            self.weights_arrays.append(new_weights)
            self.bias_arrays.append(new_bias)

    def predict(self, observation):
        self.node_layers = [observation]  # Reset node layers for each prediction
        cur_neurons = observation
        
        for index, (cur_weights, cur_bias) in enumerate(zip(self.weights_arrays, self.bias_arrays)):
            cur_neurons = np.matmul(cur_weights, cur_neurons)
            cur_neurons = np.add(cur_neurons, cur_bias)
            
            # Apply activation function
            if index != len(self.bias_arrays) - 1:
                cur_neurons = sigmoid(cur_neurons)
            else:
                cur_neurons = softmax(cur_neurons)
                
            self.node_layers.append(cur_neurons)
        return cur_neurons

    def backpropagation(self, predictions, true_label):
        num_layers = len(self.weights_arrays)
        m = 1  # batch size (1 for online learning)
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights_arrays]
        bias_gradients = [np.zeros_like(b) for b in self.bias_arrays]
        
        # Convert true label to one-hot encoding
        y = np.zeros((len(predictions), 1))
        y[int(true_label)] = 1
        
        # Output layer error
        delta = predictions - y
        
        # Backward pass
        for layer in range(num_layers - 1, -1, -1):
            if layer == num_layers - 1:
                # Output layer gradients
                weight_gradients[layer] = np.dot(delta, self.node_layers[layer].T) / m
                bias_gradients[layer] = np.sum(delta, axis=1, keepdims=True) / m
            else:
                # Hidden layer gradients
                delta = np.dot(self.weights_arrays[layer + 1].T, delta) * sigmoid_derivative(self.node_layers[layer + 1])
                weight_gradients[layer] = np.dot(delta, self.node_layers[layer].T) / m
                bias_gradients[layer] = np.sum(delta, axis=1, keepdims=True) / m
        
        # Update weights and biases
        for i in range(num_layers):
            self.weights_arrays[i] -= self.learning_rate * weight_gradients[i]
            self.bias_arrays[i] -= self.learning_rate * bias_gradients[i]

    def train_model(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = [line.strip().split(',') for line in lines[1:]]
        data = np.array(data, dtype=float)
        
        all_labels = data[:, 0]
        all_observations = data[:, 1:]
        all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])
        
        self.init_weights_and_bias()
        
        for epoch in range(self.epoch):
            total_loss = 0
            for observation, label in zip(all_observations, all_labels):
                predictions = self.predict(observation)
                loss = MCCE_Loss(label, predictions)
                total_loss += loss
                
                self.backpropagation(predictions, label)
                
            avg_loss = total_loss / len(all_observations)
            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss}")

nn = Manual_nn(7, 500, .05)

nn.train_model("xor_train.csv")

with open("xor_valid.csv", 'r') as file:
        lines = file.readlines()

data = [line.strip().split(',') for line in lines[1:]]
data = np.array(data, dtype=float)

all_labels = data[:, 0]
all_observations = data[:, 1:]
all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])


graph.plot_decision_regions(all_observations, all_labels, nn)
