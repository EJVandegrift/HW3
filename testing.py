import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def make_best_param_graph(params_dict, data_name):
    import numpy as np
    import matplotlib.pyplot as plt

    # Define specific colors for epochs
    epoch_colors = {
        10: 'red',
        25: "purple",
        50: 'blue',
        75: "pink",
        100: 'green'
    }
    possible_epochs = list(epoch_colors.keys())  # Ensure your input only has these epochs

    # Prepare the data for plotting
    x = []
    y = []
    z = []
    colors = []  # To store the color mapping
    labels = []

    for (k, epoch, rate), loss in params_dict.items():
        x.append(k)
        y.append(rate)
        z.append(loss)
        colors.append(epoch_colors[epoch])  # Map epoch to its specific color
        labels.append(f"Epoch {epoch}")

    # Convert lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(121, projection='3d')  # Left plot: 3D scatter plot

    # Plot points with preassigned colors
    scatter = ax.scatter(x, y, z, c=colors, s=50, alpha=0.8)

    # Add labels to axes
    ax.set_xlabel("k (Number of Nodes)")
    ax.set_ylabel("Learning Rate")
    ax.set_zlabel("Accuracy")
    ax.set_title(f"Accuracy vs k, Learning Rate, and Epoch on {data_name}")

    # Add a custom legend for epochs
    for epoch, color in epoch_colors.items():
        ax.scatter([], [], [], c=color, label=f"Epoch {epoch}", alpha=0.8, edgecolors='none')
    ax.legend(loc='upper right')

    # Additional 2D graphs for each independent variable
    independent_vars = ["k", "rate", "epoch"]
    independent_data = {
        "k": x,
        "rate": y,
        "epoch": np.array([epoch for (_, epoch, _) in params_dict.keys()]),
    }

    # Create subplots for 2D graphs
    fig2, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for i, var in enumerate(independent_vars):
        values = independent_data[var]
        losses = z

        # Create scatter plots for each variable
        axs[i].scatter(values, losses, alpha=0.8)
        axs[i].set_title(f"Accuracy vs {var}")
        axs[i].set_xlabel(var.capitalize())
        axs[i].set_ylabel("Accuracy")

    # Adjust layout and display both figures
    plt.tight_layout()
    plt.show()