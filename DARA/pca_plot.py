import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from matplotlib.colors import ListedColormap

from dataloader import DARA_dataset

# This function applies dimensionality reduction and plots the results
def plot_reduced_data_with_lines(dataset, label_type='category', method='PCA', components=2):
    # Extract data and labels
    all_data = [d[0].numpy() for d in dataset]  # Assuming d[0] is the data, converting tensors to numpy arrays
    all_data = np.array(all_data)  # Convert list of numpy arrays to a single numpy array

    all_labels = [d[1] for d in dataset]  # Assuming d[1] is the label

    # Apply dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=components, learning_rate='auto', init='random')
    reduced_data = reducer.fit_transform(all_data)

    # Plotting
    plt.figure(figsize=(10, 8))

    # Unique labels (different architectures)
    unique_labels = set(all_labels)

    # For each unique label (architecture), plot and connect all respective points
    for label in unique_labels:
        # Indices of points with the current label
        indices = [i for i, lbl in enumerate(all_labels) if lbl == label]
        
        # Extract respective points for this architecture
        points = reduced_data[indices, :]

        # Plot points
        plt.scatter(points[:, 0], points[:, 1], label=dataset.get_label_mapping()[label], alpha=0.6)

        # Draw lines between points: connect all points in 'points' array
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'r-', alpha=0.1)

    # Beautifying the plot
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{method} projection of the data, connected by {label_type}')
    plt.legend()
    plt.savefig(f"{method}_projection_{label_type}.png")

'''
def plot_reduced_data(dataset, label_type='category', method='PCA', components=2):
    # Extract data and labels
    all_data = [d[0].numpy() for d in dataset]  # Assuming d[0] is the data, and converting tensors to numpy arrays
    all_data = np.array(all_data)  # Convert list of numpy arrays to a single numpy array
    all_labels = [d[1] for d in dataset]  # d[1] is already an integer, representing the label
    
    # Ensure there's a valid label mapping function and apply it
    if hasattr(dataset, 'get_label_mapping'):
        all_labels_text = [dataset.get_label_mapping()[lbl] for lbl in all_labels]  # Convert numerical labels to textual labels
    else:
        all_labels_text = all_labels  # Use numerical labels as fallback

    # Apply dimensionality reduction
    if method == 'PCA':
        reducer = PCA(n_components=components)
    elif method == 'TSNE':
        reducer = TSNE(n_components=components, learning_rate='auto', init='random')
    reduced_data = reducer.fit_transform(all_data)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=all_labels, cmap='viridis', alpha=0.6)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{method} projection of the data based on {label_type}')
    plt.colorbar(scatter, ticks=range(len(set(all_labels))))
    plt.savefig(f"{method}_projection_{label_type}.png")
'''

def plot_reduced_data(dataset, label_type='category', method='PCA', components=2, is_sparse=True):
    # Extract data and labels
    all_data = np.array([d[0].numpy().flatten() for d in dataset])
    all_labels = np.array([d[1] for d in dataset])

    # Handle textual labels with LabelEncoder
    if hasattr(dataset, 'get_label_mapping'):
        label_encoder = LabelEncoder()
        all_labels_text = label_encoder.fit_transform([dataset.get_label_mapping()[lbl] for lbl in all_labels])
        label_names = label_encoder.classes_
        ticks = range(len(label_names))
    else:
        all_labels_text = all_labels
        label_names = sorted(set(all_labels))
        ticks = range(len(label_names))

    # Dimensionality reduction
    if method == 'PCA':
        if is_sparse:
            reducer = SparsePCA(n_components=components, alpha=1)  # alpha is the sparsity controlling parameter
        else:
            reducer = PCA(n_components=components)
    elif method == 'TSNE':
        # For TSNE, you typically don't use SparsePCA directly because TSNE is not designed for sparse data.
        # However, you might reduce dimensionality first using SparsePCA if data is really high-dimensional and sparse.
        if is_sparse:
            pre_reducer = SparsePCA(n_components=50, alpha=1)  # Reduce dimensionality before applying TSNE
            all_data = pre_reducer.fit_transform(all_data)
            reducer = TSNE(n_components=components, learning_rate='auto', init='random')
        else:
            reducer = TSNE(n_components=components, learning_rate='auto', init='random')
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    reduced_data = reducer.fit_transform(all_data)

    # Plot
    color_list = [
        'red', 'blue', 'green', 'orange', 'purple', 
        'brown', 'pink', 'gray', 'olive', 'cyan',
        'darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet', 
        'sienna', 'lightpink', 'lightgray', 'lime', 'skyblue', 
        'gold', 'teal', 'coral', 'navy', 'magenta', 
        'yellowgreen', 'lavender', 'maroon', 'aqua', 'chocolate', 
        'steelblue', 'fuchsia', 'crimson', 'forestgreen', 'indigo', 
        'darkturquoise', 'goldenrod', 'mediumseagreen', 'tomato', 'slateblue'
    ]
    if len(set(all_labels_text)) > len(color_list):
        print("Warning: Not enough colors specified for the number of categories. Consider adding more colors.")

    # Create a colormap from the list of colors
    cmap = ListedColormap(color_list[:len(set(all_labels_text))])

    plt.figure(figsize=(8, 6))

    # Use a colormap for scatter points
    # cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(all_labels_text), vmax=max(all_labels_text))

    # Plot lines for each category
    for label in sorted(set(all_labels_text)):
        indices = np.where(all_labels_text == label)
        category_data = reduced_data[indices]
        # Use the normalized color for lines corresponding to the label
        plt.plot(category_data[:, 0], category_data[:, 1], '-o', alpha=0.5, color=cmap(norm(label)), label=label_names[label] if hasattr(dataset, 'get_label_mapping') else str(label))
            
    # Then plot all points as scatter. This time we don't need individual labels since the lines already have them.
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=all_labels_text, cmap=cmap, alpha=0.6)

    # Adding colorbar
    cbar = plt.colorbar(scatter, ticks=ticks)
    cbar.set_ticklabels(label_names)

    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(f'{method} projection of the data based on {label_type}')

    # Here's the fix: Create a custom legend for the lines only, as the colorbar already explains the scatter.
    # This eliminates the dual legend issue by not calling plt.legend() after plotting the lines.
    # Instead, you could explicitly define a legend for either lines or scatter points if needed, but avoid redundancy.

    plt.savefig(f"{method}_projection_{label_type}.png")

# Example usage with your data
if __name__ == "__main__":
    # Load your dataset
    vec_path = './data_cleaned_filtered.json'
    full_dataset = DARA_dataset(dict_path=vec_path, label_type="model_type")  # or "model_type" or "task"
    plot_reduced_data(full_dataset, label_type="model_type", method='PCA', components=2)
    plot_reduced_data(full_dataset, label_type="model_type", method='TSNE', components=2)

    full_dataset = DARA_dataset(dict_path=vec_path, label_type="arch")  # or "model_type" or "task"
    plot_reduced_data(full_dataset, label_type="arch", method='PCA', components=2)
    plot_reduced_data(full_dataset, label_type="arch", method='TSNE', components=2)

    # Plotting the data reduced by PCA
    # plot_reduced_data_with_sequential_lines(full_dataset, label_type="task", method='PCA', components=2)

    # Plotting the data reduced by t-SNE
    # plot_reduced_data_with_sequential_lines(full_dataset, label_type="task", method='TSNE', components=2)



    # # Plotting the data reduced by PCA
    # plot_reduced_data_with_lines(full_dataset, label_type="task", method='PCA', components=2)

    # # Plotting the data reduced by t-SNE
    # plot_reduced_data_with_lines(full_dataset, label_type="task", method='TSNE', components=2)
