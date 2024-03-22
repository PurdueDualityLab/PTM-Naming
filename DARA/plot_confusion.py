import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns  # Seaborn is used for heatmap visualization


def plot_confusion_matrix_heatmap(preds, targets, index_to_label, label_type):
    # Calculate the confusion matrix
    cm = confusion_matrix(targets, preds)
    # Normalize the confusion matrix by row (i.e., by the number of samples in each true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))  # Adjust size as needed
    
    labels_ordered = [index_to_label[str(i)] for i in range(len(index_to_label))]
    
    # Plot the heatmap without annotations but with grid lines
    ax = sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=labels_ordered, yticklabels=labels_ordered, linewidths=.5)
    
    # Decrease the font size of the category labels for both axes
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)  # Adjust fontsize for x-axis labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)  # Adjust fontsize for y-axis labels
    
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_heatmap_{label_type}.png", dpi=300)


# Open the file and read the data
with open('index_to_label.json', 'r') as f:
    index_to_label = json.load(f)

with open('label_type.json', 'r') as f:
    label_type = json.load(f)

# Load the predictions and targets
all_fold_preds = np.load('all_fold_preds.npy')
all_fold_targets = np.load('all_fold_targets.npy')

plot_confusion_matrix_heatmap(all_fold_preds, all_fold_targets, index_to_label, label_type)