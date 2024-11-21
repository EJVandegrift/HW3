import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)

class NeuralNetworkWrapper:
    """Wrapper class to make Manual_nn compatible with the plotting functions"""
    def __init__(self, model):
        self.model = model
    
    def predict(self, X):
        """
        Adapts the model's predict function to work with the plotting code
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            numpy array of predictions
        """
        predictions = []
        for sample in X:
            # Reshape input to (2, 1) as expected by the model
            sample_reshaped = sample.reshape(2, 1)
            # Get prediction and convert to class label
            pred = self.model.predict(sample_reshaped)
            predictions.append(np.argmax(pred))
        return np.array(predictions)
def plot_decision_regions(X, y, model, resolution=0.01):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    
    # Setup colors
    point_colors = ('red', 'blue')
    region_colors = ('red', 'blue')
    decision_cmap = ListedColormap(region_colors)
    
    # Convert 3D array to 2D for plotting
    X_2d = np.array([[x[0][0], x[1][0]] for x in X])
    
    # Calculate number of rows needed (2 columns)
    num_plots = model.k + 1  # Composite + individual nodes
    num_rows = (num_plots + 2) // 3  # Ceiling division to handle odd numbers
    
    # Create figure with subplots in 2 columns
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    # Convert axs to 1D array for easier indexing
    axs = axs.flatten()
    
    # If odd number of plots, remove the last (empty) subplot
    if num_plots % 2 != 0:
        fig.delaxes(axs[-1])
    
    # Calculate grid bounds with smaller padding
    padding = 0.5
    x1_min, x1_max = X_2d[:, 0].min() - padding, X_2d[:, 0].max() + padding
    x2_min, x2_max = X_2d[:, 1].min() - padding, X_2d[:, 1].max() + padding
    
    # Create mesh grid with increased resolution (fewer points)
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    
    # Prepare all mesh points at once
    mesh_points = np.vstack((xx1.ravel(), xx2.ravel()))
    
    def batch_predict(points, batch_size=1000):
        predictions = []
        for i in range(0, points.shape[1], batch_size):
            batch = points[:, i:i+batch_size]
            batch_predictions = []
            for j in range(batch.shape[1]):
                pred = model.predict(batch[:, j:j+1])
                batch_predictions.append(pred)
            predictions.extend(batch_predictions)
        return np.array(predictions)
    
    plots = ['Composite'] + [f'Node {i}' for i in range(model.k)]
    
    for idx, ax in enumerate(plots):
        if idx == 0:  # Composite decision boundary
            # Make predictions in batches
            predictions = batch_predict(mesh_points)
            Z = np.array([np.argmax(pred) for pred in predictions])
        else:  # Individual node boundaries
            # Calculate hidden layer activations in batches
            Z = np.zeros(mesh_points.shape[1])
            for i in range(0, mesh_points.shape[1], 1000):
                batch = mesh_points[:, i:i+1000]
                activations = np.matmul(model.weights_arrays[0], batch)
                activations = np.add(activations, model.bias_arrays[0])
                activations = sigmoid(activations)
                Z[i:i+1000] = activations[idx-1, :] > 0.5
        
        Z = Z.reshape(xx1.shape)
        
        # Plot contours
        axs[idx].contourf(xx1, xx2, Z, alpha=0.2, cmap=decision_cmap)
        axs[idx].set_xlim(xx1.min(), xx1.max())
        axs[idx].set_ylim(xx2.min(), xx2.max())
        
        # Plot class samples
        for cl_idx, cl in enumerate(np.unique(y)):
            axs[idx].scatter(
                x=X_2d[y == cl, 0],
                y=X_2d[y == cl, 1],
                alpha=1,
                c=point_colors[cl_idx],
                label=cl
            )
        
        axs[idx].set_xlabel('Feature 1')
        axs[idx].set_ylabel('Feature 2')
        axs[idx].set_title(plots[idx])
        
        if idx > 0:
            legend_elements = [
                Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=region_colors[1], label='1.0',
                      markersize=10),
                Line2D([0], [0], marker='s', color='w',
                      markerfacecolor=region_colors[0], label='0.0',
                      markersize=10)
            ]
            axs[idx].legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()

def plot_node_activations(model, features, targets, title="Hidden Node Activations"):
    """
    Visualizes the activation patterns of each hidden node in the network.
    
    Args:
        model: Manual_nn model instance
        features: numpy array of shape (n_samples, n_features)
        targets: numpy array of shape (n_samples,)
        title: plot title
    """
    n_nodes = model.k
    fig, axes = plt.subplots(1, n_nodes + 1, figsize=(4 * (n_nodes + 1), 4))
    fig.suptitle(title)
    
    # Plot decision boundary in first subplot
    plot_decision_regions(
        features, targets, model,
        axis=axes[0],
        title='Overall Decision Boundary'
    )
    
    # Define grid for visualization
    min1, max1 = features[:, 0].min() - 1, features[:, 0].max() + 1
    min2, max2 = features[:, 1].min() - 1, features[:, 1].max() + 1
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    xx, yy = np.meshgrid(x1grid, x2grid)
    
    # Get activations for each point in the grid
    grid_points = []
    activations = []
    for x1, x2 in zip(xx.flatten(), yy.flatten()):
        input_point = np.array([[x1], [x2]])
        model.predict(input_point)  # This updates node_layers
        # Get hidden layer activations (index 1 for first hidden layer)
        activation = model.node_layers[1]
        activations.append(activation.flatten())
        grid_points.append([x1, x2])
    
    activations = np.array(activations)
    
    # Plot activation of each hidden node
    for i in range(n_nodes):
        node_activations = activations[:, i].reshape(xx.shape)
        im = axes[i + 1].contourf(xx, yy, node_activations, cmap='coolwarm', levels=20)
        axes[i + 1].set_title(f'Node {i + 1} Activation')
        plt.colorbar(im, ax=axes[i + 1])
        
        # Plot training points
        for label in [0, 1]:
            mask = targets == label
            axes[i + 1].scatter(
                features[mask, 0],
                features[mask, 1],
                alpha=0.5,
                s=20,
                c='black',
                marker='.' if label == 0 else '^'
            )
    
    plt.tight_layout()
    return fig, axes