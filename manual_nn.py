import numpy as np
import random
import math

# IMPORTANT:
# If there are three possible classifications (e.g. red, yellow, blue), and the true label is yellow,
# 'true_labels' should be [0, 1, 0] with predictions indexes corresponding to the same indexes of true_labels
def MCCE_Loss(true_label, predictions, num_dif_labels=2):
    '''
    MCCE Loss for a single observatoin

    :param true_labels: Hot-encoded array of labels (only single element should be 1)
    :param predictions: Array of models predictions in the form of percentages (sums to 1)
    :param num_dif_labels:
    :return:
    '''

    # Make sure predictions are normal lists
    if hasattr(predictions, 'tolist'):  # Check if predictions is a NumPy array
        predictions = predictions.flatten().tolist()

    hot_encoded_labels = [1 if true_label == i else 0 for i in range(num_dif_labels)]
    total = 0

    for i in range(num_dif_labels):
        if predictions[i] == 0:
            predictions[i] = 1e-15
        total += hot_encoded_labels[i] * math.log(predictions[i])

    return total * -1


def sigmoid(z):
    return 1/(1 + np.exp(-z))

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)

class Manual_nn():
    def __init__(self, k, epoch, learning_rate, num_hid_layers=1):
        self.k = k
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.num_hid_layers = num_hid_layers
        
        # IMPORTANT: This doesn't include the input layer 
        #because we don't derive it from a calculation, it is given, therefore it's not needed for back propagation
        self.node_layers = [] 
        self.weights_arrays = []
        self.bias_arrays = []
        

    # randomizes weights
    def init_weights_and_bias(self, num_features=2, num_outputs=2):
        self.weights_arrays = []
        self.bias_arrays = []
        # Need num_hidden_layers + 1 sets of weights
        for i in range(self.num_hid_layers + 1):
            # weight matrix dimensions = (num_nodes_next_layer, num_nodes_prev_layer)
            # Randomize weights for unbiased initialization

            # First iteration requires knowing num features in observation
            if(i == 0):
                new_weights = np.random.uniform(-.1, .1, size=(self.k, num_features))
                self.weights_arrays.append(new_weights)
                # Unless observation data is multidimensional, the num cols should be 1
                new_bias = np.random.uniform(-.1, .1, size=(new_weights.shape[0], 1))
                self.bias_arrays.append(new_bias)
                
            # Last iteration requires knowing how many outputs (classifications) there are
            elif(i == self.num_hid_layers):
                new_weights = np.random.uniform(-.1, .1, size=(num_outputs, self.k))
                self.weights_arrays.append(new_weights)
                
                new_bias = np.random.uniform(-.1, .1, size=(num_outputs, 1))
                self.bias_arrays.append(new_bias)

            # Intermediate layers will simply be kxk due to all hidden layers having the same num nodes
            else:
                self.weights_arrays.append(np.random.rand(self.k, self.k))
                self.bias_arrays.append(np.random.rand(self.k, self.k))


    def predict(self, observation):
        self.node_layers.append(observation)
        cur_neurons = observation
        # Until at output, pass data through neural network each weight and bias at a time
        for index, (cur_weights, cur_bias) in enumerate(zip(self.weights_arrays, self.bias_arrays)):
            
            cur_neurons = np.matmul(cur_weights, cur_neurons) # multiply weights and features
            cur_neurons = np.add(cur_neurons, cur_bias) # add bias term

            # Need to regularize data -> if in hidden node use sigmoid, if at output, use softmax
            cur_neurons = sigmoid(cur_neurons) if (index != len(self.bias_arrays) - 1) else softmax(cur_neurons)
            
            # Need each node value for layers for back propagation
            self.node_layers.append(cur_neurons)
        return cur_neurons
    
    
    def get_new_weights_at_layer(self, weight_layer_pos, predictions, true_label):
        """Adjusts a single layer of weights

        Args:
            weight_layer_pos (int): layer position (index)
            predictions (np.array): prediction from model given
            true_label (_type_): _description_
        """
        weight_matrix = self.weights_arrays[weight_layer_pos]
        
        prev_nodes = self.node_layers[weight_layer_pos]
        next_nodes = self.node_layers[weight_layer_pos + 1]
        
        # flattened predictoins for ease of indexing
        one_dim_predicts = predictions.flatten().tolist()
        one_dim_nodes = prev_nodes.flatten().tolist()
        
        hot_encoded_labels = [1 if true_label == i else 0 for i in range(len(one_dim_predicts))]
        
        # Ensure weights calculated before others don't get included in the next weights calculations
        weights_dimensions = weight_matrix.shape
        new_weights = np.zeros(weights_dimensions)
        
        # assuming we're dealing with 2d arrays
        for row in range(weights_dimensions[0]):
            for col in range(weights_dimensions[1]):
                if weight_layer_pos == 0:
                    sigma = 0
                    # need the next layer of weights for the gradient descent
                    next_weight_layer = self.weights_arrays[weight_layer_pos + 1]
                    
                    for output_idx in range(len(hot_encoded_labels)):
                        # Needed weights from next layer correspond to the column of the current weight
                        sigma += (one_dim_predicts[output_idx] - hot_encoded_labels[output_idx]) * next_weight_layer[output_idx, row]
                        
                    hidden_node_val = next_nodes[row]
                    
                    error_signal = sigma * hidden_node_val * (1-hidden_node_val)
                    
                    gradient = error_signal * prev_nodes[col]
                    
                    old_weight = weight_matrix[row, col]
                    new_weights[row, col] = old_weight - self.learning_rate * gradient
                    
                elif weight_layer_pos == 1:
                    prediction_prob = one_dim_predicts[row]
                    corresponding_label = hot_encoded_labels[row]
                    prediction_prob = one_dim_predicts[row]
                    corresponding_label = hot_encoded_labels[row]
                    prev_node_val = one_dim_nodes[col]

                    gradient = (prediction_prob - corresponding_label) * prev_node_val
                    old_weight = weight_matrix[row, col]
                    
                    new_weights[row, col] = old_weight - self.learning_rate * gradient
                
        
        # replace old weights with new ones
        return new_weights
                
    
    def adjust_all_weights(self, predictions, true_label):
        new_weights = []
        for i in range(len(self.weights_arrays) - 1, -1, -1):
            new_weights.append(self.get_new_weights_at_layer(i, predictions, true_label))

        new_weights.reverse()
        self.weights_arrays = new_weights
        

    def train_model(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Parse the data
        data = [line.strip().split(',') for line in lines[1:]]  # Skip the header and split each line
        data = np.array(data, dtype=float)  # Convert to a NumPy array with float type

        # Separate labels and observations using slicing
        all_labels = data[:, 0]  # First column is the labels
        all_observations = data[:, 1:]  # Remaining columns are the features

        all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])
        # Print th
        
        
        self.init_weights_and_bias()
        for i in range(self.epoch):
            for observation, label in zip(all_observations, all_labels):
                prediction_probs = self.predict(observation)
                
                # print(f"Loss: {MCCE_Loss(label, prediction_probs)}")
                
                self.adjust_all_weights(prediction_probs, label)
           
            
    def test_model(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        # Parse the data
        data = [line.strip().split(',') for line in lines[1:]]  # Skip the header and split each line
        data = np.array(data, dtype=float)  # Convert to a NumPy array with float type

        # Separate labels and observations using slicing
        all_labels = data[:, 0]  # First column is the labels
        all_observations = data[:, 1:]  # Remaining columns are the features

        all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])
        num_guessed_correctly = 0
        
        for observation, label in zip(all_observations, all_labels):
            prediction_probs = self.predict(observation).flatten().tolist()
            
            highest_prob = max(prediction_probs)
            guessed_classification = prediction_probs.index(highest_prob)

            if guessed_classification == label:
                num_guessed_correctly += 1
            # print(f"{prediction_probs} | {label}")
        return num_guessed_correctly / len(all_observations)
        
        

nn = Manual_nn(3, 30, .01)

# testing
nn.train_model("xor_valid.csv")
print(nn.test_model("xor_test.csv"))




