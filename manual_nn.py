import numpy as np
import random
import math
import graph
import matplotlib.pyplot as plt
def MCCE_Loss(true_label, predictions, num_dif_labels=2):
    '''
    MCCE Loss for a single observation

    :param true_labels: Hot-encoded array of labels (only single element should be 1)
    :param predictions: Array of models predictions in the form of percentages (sums to 1)
    :param num_dif_labels: Number of different possible classifications
    :return: Loss value
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
        # ensure arrays start empty
        self.weights_arrays = []
        self.bias_arrays = []
        
        # Use Xavier Glorot to reduce variance for intial values
        for i in range(self.num_hid_layers + 1):
            if(i == 0):
                # Input to first hidden layer
                scale = np.sqrt(2.0 / (num_features + self.k))
                new_weights = np.random.normal(0, scale, size=(self.k, num_features))
                new_bias = np.zeros((self.k, 1))  # Initialize biases to zero
                
            elif(i == self.num_hid_layers):
                # Hidden to output layer
                scale = np.sqrt(2.0 / (self.k + num_outputs))
                new_weights = np.random.normal(0, scale, size=(num_outputs, self.k))
                new_bias = np.zeros((num_outputs, 1))
                
            else:
                # Hidden to hidden layer (if you had more than 1 hidden layer)
                scale = np.sqrt(2.0 / (self.k + self.k))
                new_weights = np.random.normal(0, scale, size=(self.k, self.k))
                new_bias = np.zeros((self.k, 1))
                
            self.weights_arrays.append(new_weights)
            self.bias_arrays.append(new_bias)

    def predict(self, observation):
        self.node_layers = [observation]  
        cur_neurons = observation
        
        # Pass values through nn, each weight and bias at a time(forward pass)
        for index, (cur_weights, cur_bias) in enumerate(zip(self.weights_arrays, self.bias_arrays)):
            cur_neurons = np.matmul(cur_weights, cur_neurons)
            cur_neurons = np.add(cur_neurons, cur_bias)
            
            # Apply activation function - use sigmoid if hidden layer, othewise softmax
            cur_neurons = sigmoid(cur_neurons) if (index != len(self.bias_arrays) - 1) else softmax(cur_neurons)
            self.node_layers.append(cur_neurons)
            
        return cur_neurons
    
    def get_new_weights_and_biases_at_layer(self, weight_layer_pos, predictions, true_label):
        weight_matrix = self.weights_arrays[weight_layer_pos]
        bias_vector = self.bias_arrays[weight_layer_pos]
        
        prev_nodes = self.node_layers[weight_layer_pos]
        next_nodes = self.node_layers[weight_layer_pos + 1]
        
        one_dim_predicts = predictions.flatten().tolist()
        one_dim_nodes = prev_nodes.flatten().tolist()
        
        hot_encoded_labels = [1 if true_label == i else 0 for i in range(len(one_dim_predicts))]
        
        weights_dimensions = weight_matrix.shape

        # need to make copies of matrices because this function only calculates a single layer at a time
        new_weights = np.zeros(weights_dimensions)
        new_biases = np.zeros_like(bias_vector)
        
        # Iterate over each value in weight/bias layer individually
        for row in range(weights_dimensions[0]):
            for col in range(weights_dimensions[1]):
                if weight_layer_pos == 0:
                    # Hidden layer
                    sigma = 0
                    next_weight_layer = self.weights_arrays[weight_layer_pos + 1]
                    
                    for output_idx in range(len(hot_encoded_labels)):
                        sigma += (one_dim_predicts[output_idx] - hot_encoded_labels[output_idx]) * next_weight_layer[output_idx, row]
                    
                    hidden_node_val = next_nodes[row]
                    error_signal = sigma * sigmoid_derivative(hidden_node_val)
                    
                    # weight update
                    gradient = error_signal * prev_nodes[col]
                    old_weight = weight_matrix[row, col]
                    new_weights[row, col] = old_weight - self.learning_rate * gradient
                    
                    # wias update 
                    if col == 0:
                        new_biases[row] = bias_vector[row] - self.learning_rate * error_signal
                    
                elif weight_layer_pos == 1:
                    # output layer
                    prediction_prob = one_dim_predicts[row]
                    corresponding_label = hot_encoded_labels[row]
                    prev_node_val = one_dim_nodes[col]

                    # weight update
                    gradient = (prediction_prob - corresponding_label) * prev_node_val
                    old_weight = weight_matrix[row, col]
                    new_weights[row, col] = old_weight - self.learning_rate * gradient
                    
                    # bias update
                    if col == 0:
                        bias_gradient = prediction_prob - corresponding_label
                        new_biases[row] = bias_vector[row] - self.learning_rate * bias_gradient
        
        return new_weights, new_biases
    
    def adjust_all_weights(self, predictions, true_label):
        # Need to get each layer individually - avoid updating model before all calcs are done
        new_weights = []
        new_biases = []
        # Start at the last layer of nm - doesn't make a difference computationally, but conceptually it makes more sense
        for i in range(len(self.weights_arrays) - 1, -1, -1):
            weights, biases = self.get_new_weights_and_biases_at_layer(i, predictions, true_label)
            new_weights.append(weights)
            new_biases.append(biases)

        # Append adds values to the end of the list, so layers are being stored in reverse order
        # To ensure the dimensions of the weights and matrices are aligned, reverse the lists
        new_weights.reverse()
        new_biases.reverse()

        # have all the weights and biases - and in the correct layer order - so can finally update model
        self.weights_arrays = new_weights
        self.bias_arrays = new_biases

    def train_model(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = [line.strip().split(',') for line in lines[1:]]
        data = np.array(data, dtype=float)

        all_labels = data[:, 0]
        all_observations = data[:, 1:]
        all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])
        
        # Ensure the weights and biases are intialized before training
        # if already populated from prior training, it will simply pass
        if len(self.weights_arrays) == 0 or len(self.bias_arrays) == 0:
            self.init_weights_and_bias()
            

        # train model as many times as epoch indicates
        for i in range(self.epoch):
            total_loss = 0
            for observation, label in zip(all_observations, all_labels):
                prediction_probs = self.predict(observation)
                total_loss += MCCE_Loss(label, prediction_probs)
                self.adjust_all_weights(prediction_probs, label)
            
            avg_loss = total_loss / len(all_observations)
            # if i % 5 == 0:  # Print every 5 epochs
            #     print(f"Epoch {i}, Average Loss: {avg_loss}")
    
    def test_model(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()

        data = [line.strip().split(',') for line in lines[1:]]
        data = np.array(data, dtype=float)

        all_labels = data[:, 0]
        all_observations = data[:, 1:]
        all_observations = np.array([obs.reshape(2, 1) for obs in all_observations])
        
        num_guessed_correctly = 0
        total_loss = 0
        
        for observation, label in zip(all_observations, all_labels):
            prediction_probs = self.predict(observation)
            total_loss += MCCE_Loss(label, prediction_probs)
            
            highest_prob = max(prediction_probs.flatten().tolist())
            guessed_classification = prediction_probs.flatten().tolist().index(highest_prob)

            if guessed_classification == label:
                num_guessed_correctly += 1
                
        accuracy = num_guessed_correctly / len(all_observations)
        avg_loss = total_loss / len(all_observations)
        
        # print(f"Test Accuracy: {accuracy:.4f}")
        # print(f"Average Test Loss: {avg_loss:.4f}")
        
        return accuracy * 100
    

    def train_model_loss(self, train_file, valid_file, data_set):
        # load training data
        with open(train_file, 'r') as file:
            lines = file.readlines()
        train_data = [line.strip().split(',') for line in lines[1:]]
        train_data = np.array(train_data, dtype=float)
        train_labels = train_data[:, 0]
        train_observations = np.array([obs.reshape(2, 1) for obs in train_data[:, 1:]])
        
        with open(valid_file, 'r') as file:
            lines = file.readlines()
        valid_data = [line.strip().split(',') for line in lines[1:]]
        valid_data = np.array(valid_data, dtype=float)
        valid_labels = valid_data[:, 0]
        valid_observations = np.array([obs.reshape(2, 1) for obs in valid_data[:, 1:]])
        
        self.init_weights_and_bias()
        
        training_losses = []
        validation_losses = []
        
        for i in range(self.epoch):
            total_train_loss = 0
            for observation, label in zip(train_observations, train_labels):
                prediction_probs = self.predict(observation)
                total_train_loss += MCCE_Loss(label, prediction_probs)
                self.adjust_all_weights(prediction_probs, label)
            
            avg_train_loss = total_train_loss / len(train_observations)
            training_losses.append(avg_train_loss)
            
            total_valid_loss = 0
            for observation, label in zip(valid_observations, valid_labels):
                prediction_probs = self.predict(observation)
                total_valid_loss += MCCE_Loss(label, prediction_probs)
            
            avg_valid_loss = total_valid_loss / len(valid_observations)
            validation_losses.append(avg_valid_loss)
            
            if i % 5 == 0:
                print(f"Epoch {i}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(self.epoch), training_losses, label='Training Loss')
        plt.plot(range(self.epoch), validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{data_set} Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()


def find_best_params(k_list, epoch_list, learning_rate_list, train_file, valid_file):
    param_results = {}
    num_combinations = len(k_list) * len(epoch_list) * len(learning_rate_list)
    i = 1

    for k in k_list:
        for epoch in epoch_list:
            for rate in learning_rate_list:
                model = Manual_nn(k, epoch, rate)
                model.train_model(train_file)

                param_results[(k, epoch, rate)] = model.test_model(valid_file)

                print(f"Percent done: {round((i / num_combinations) * 100, 2)}%")
                i += 1

    # Done with getting results -> order from best to worst results
    results_list = list(param_results.items())

    # Make analyzing results easier with order from best to worst accuracy
    return sorted(results_list, key=lambda x: x[1], reverse=True)

# Example usage
if __name__ == "__main__":

    nn = Manual_nn(k=7, epoch=100, learning_rate=0.01)
    nn.train_model("xor_train.csv")
    print(nn.test_model("xor_test.csv"))
    nn.train_model("xor_train.csv")
    print(nn.test_model("xor_test.csv"))