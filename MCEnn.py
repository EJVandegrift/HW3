
import torch
from torch import nn
import pandas as pd
import graph
import pandas as pd
import numpy as np
import testing

def find_optimal_params(k_list, epoch_list, learning_list, data_file_list):
    file_to_params = {}
    all_results = {}
    for file in data_file_list:

        k_epoch_learning = {}
        for k in k_list:
            for epoch in epoch_list:
                for rate in learning_list:
                    k_epoch_learning[(k, epoch, rate)] = None

        cur_train_data = pd.read_csv(file)
        labels = cur_train_data.iloc[:, 0]
        features = cur_train_data.iloc[:, 1:]

        # Convert to PyTorch tensors
        labels_tensor = torch.tensor(labels.to_numpy(), dtype=torch.long)
        features_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)

        train_tensor = [[label, feature] for label, feature in zip(labels_tensor, features_tensor)]

        for k in k_list:
            model = MCE_FF(k=k)

            # Loss function (multi-class cross entropy) and optimizer
            criterion = nn.CrossEntropyLoss()

            for rate in learning_list:
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=rate)  # Low learning rate to not over estimate changes (long to train)

                # Training loop
                for epoch in epoch_list:
                    for rising_epoch in range(epoch):
                        total_loss = 0.0
                        for item in train_tensor:
                            feature = item[1].unsqueeze(0)
                            actual_result = torch.tensor([item[0]])

                            # Forward pass
                            prediction = model(feature)

                            # Compute loss - implicitly applies softmax and calculates magnitude of error relative to actual label
                            loss = criterion(prediction, actual_result)

                            # Backward pass and optimization
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()
                        if (rising_epoch == epoch - 1):
                            k_epoch_learning[(k, epoch, rate)] = total_loss / len(train_tensor)
                            print(
                                f"(k={k}, epoch={epoch}, learning_rate={rate}) -> Final loss: {total_loss / len(train_tensor)}\n")

        # best_parameters = min(k_epoch_learning, key=k_epoch_learning.get)
        # file_to_params[file] = (best_parameters, min(k_epoch_learning.values()))
        all_results[file] = k_epoch_learning
    return all_results

def make_MCE_FF(k, learning_rate, epoch, file_name):
    cur_train_data = pd.read_csv(file_name)
    labels = cur_train_data.iloc[:, 0]
    features = cur_train_data.iloc[:, 1:]

    # Convert to PyTorch tensors
    labels_tensor = torch.tensor(labels.to_numpy(), dtype=torch.long)
    features_tensor = torch.tensor(features.to_numpy(), dtype=torch.float32)

    train_tensor = [[label, feature] for label, feature in zip(labels_tensor, features_tensor)]

    model = MCE_FF(k=k)

    # Loss function (multi-class cross entropy) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)  # Low learning rate to not over estimate changes (long to train)

    for rising_epoch in range(epoch):
        total_loss = 0.0
        for item in train_tensor:
            feature = item[1].unsqueeze(0)
            actual_result = torch.tensor([item[0]])

            # Forward pass
            prediction = model(feature)

            # Compute loss - implicitly applies softmax and calculates magnitude of error relative to actual label
            loss = criterion(prediction, actual_result)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model

class MCE_FF(torch.nn.Module):

    def __init__(self, k):
        super(MCE_FF, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=k),  # Hidden layer
            nn.Sigmoid()  # Activation function
        )
        # Define the output layer (no activation)
        self.output_layer = nn.Linear(in_features=k, out_features=2)  # Output layer for 2 classes

    def forward(self, x):
        # Pass through the hidden layer
        x = self.hidden_layer(x)
        # Pass through the output layer
        x = self.output_layer(x)
        return x

    def predict(self, x):
        # Ensure the input is a tensor
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        elif not torch.is_tensor(x):
            raise ValueError("Input must be a PyTorch tensor or NumPy array.")

        # Perform forward pass
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)

        # Return as NumPy array for compatibility with plotting
        return predicted_class.detach().cpu().numpy()

# file_info = find_optimal_params([2, 3, 5, 7, 9], [10, 25, 50, 75, 100], [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1], ["xor_train.csv", "spiral_train.csv"])
file_info = find_optimal_params([7, 9], [75, 100], [.09, .1], ["xor_train.csv", "spiral_train.csv"])

for file in file_info:
    testing.make_best_param_graph(file_info[file], file)

# file_name = list(file_info.keys())[0]
# best_params = file_info[file_name][0]
#
# print(best_params)
#
# model = make_MCE_FF(k=best_params[0], epoch=best_params[1], learning_rate=best_params[2], file_name=file_name)
#
#
# cur_train_data = pd.read_csv("xor_test.csv")
# labels = cur_train_data.iloc[:, 0]
# features = cur_train_data.iloc[:, 1:]
#
# features = features.to_numpy()
# labels = labels.to_numpy()
#
# graph.plot_decision_regions(features, labels, model, axis=None, title="Test")
# graph.plt.show()