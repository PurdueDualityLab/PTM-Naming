import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import seaborn as sns  # Seaborn is used for heatmap visualization

random.seed(42)

def plot_confusion_matrix_heatmap(preds, targets, index_to_label, label_type, label_to_domain):
    if label_type == "arch":
        # Randomly select 30 values for each key in label_to_domain
        selected_labels = {}
        for domain, labels in label_to_domain.items():
            if len(labels) > 30:
                selected_labels[domain] = sorted(random.sample(labels, 30))
            else:
                selected_labels[domain] = sorted(labels)

        # Reassign label_to_domain to only include the selected labels
        label_to_domain = selected_labels
        selected_labels_set = {label for labels in selected_labels.values() for label in labels}
        
        # Create a new mapping from label to index and index to label
        label_to_index = {label: idx for idx, label in index_to_label.items() if label in selected_labels_set}
        index_to_label = {idx: label for idx, label in index_to_label.items() if label in selected_labels_set}
        
        # Filter preds and targets to only include the selected labels
        filtered_preds = []
        filtered_targets = []
        for pred, target in zip(preds, targets):
            if str(pred) in label_to_index.values() and str(target) in label_to_index.values():
                filtered_preds.append(str(pred))
                filtered_targets.append(str(target))

        preds = np.array(filtered_preds)
        targets = np.array(filtered_targets)
        
        # Filter out the labels that are not in the filtered targets
        index_to_label = {idx: label for idx, label in index_to_label.items() if idx in set(filtered_targets)}
        label_to_index = {label: idx for idx, label in index_to_label.items()}
        label_to_domain = {domain: [label for label in labels if label in label_to_index] for domain, labels in label_to_domain.items()}
            
    labels_ordered = []
    for domain, models in label_to_domain.items():
        labels_ordered.extend(models)
    # Ensure all labels are included
    labels_ordered = [label for label in labels_ordered if label in index_to_label.values()]

    if label_type == "task":
        count = 0
        filtered_targets = []
        filtered_preds = []
        for t, p in zip(targets, preds):
            if isinstance(t, (list, np.ndarray)) and sum(t) == 1:
                filtered_targets.append(np.argmax(t))
                filtered_preds.append(np.argmax(p))
            else:
                count += 1
        targets = np.array(filtered_targets)
        preds = np.array(filtered_preds)
        print(f"Number of samples with multiple labels: {count}")
    targets = [index_to_label[str(target)] for target in targets]
    preds = [index_to_label[str(pred)] for pred in preds]

    
    used_labels = set(targets) | set(preds)

    # Filter labels_ordered to include only used labels
    labels_ordered = [label for label in labels_ordered if label in used_labels]
    # Calculate the confusion matrix 
    cm = confusion_matrix(targets, preds, labels=labels_ordered)

    # Normalize the confusion matrix by row (i.e., by the number of samples in each true class)
    # cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    with np.errstate(all='ignore'):
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    
    plt.figure(figsize=(10, 7))  # Adjust size as needed
    # Plot the heatmap without annotations but with grid lines
    ax = sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=labels_ordered, yticklabels=labels_ordered, linewidths=.5)
    
    # Decrease the font size of the category labels for both axes
    fontsize = 10 if label_type == 'task' else 5    # Adjust fontsize for x-axis labels # model_type: 5, task:10
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    ax.tick_params(axis='both', which='both', length=0)
    
    # Add domain labels below the x-axis
    domain_positions = {}
    current_position = 0
    for domain, models in label_to_domain.items():
        filtered_models = [m for m in models if m in labels_ordered]
        if not filtered_models:
            continue
        start = current_position
        end = current_position + len(filtered_models) - 1
        domain_positions[domain] = (start, end)
        current_position = end + 1
    
    for i, (domain, (start, end)) in enumerate(domain_positions.items()):
        if domain == "Multimodal" and label_type == "task":
            ax.text((start + end+1) / 2, len(labels_ordered)-0.1, domain, ha='center', va='bottom', fontsize=8, color='black', rotation=90) # fontsize - model_type:8, task:10
            ax.text(1.5, (start + end + 1) / 2, domain, ha='center', va='center', fontsize=8, color='black')
        else:
            ax.text((start + end+1) / 2, len(labels_ordered)-0.1, domain, ha='center', va='bottom', fontsize=8, color='black')
            ax.text(0.5, (start + end + 1) / 2, domain, ha='center', va='center', fontsize=8, color='black', rotation=90)
        if i < len(domain_positions) - 1:
            ax.axvline(x=end+1, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(y=end+1, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"Naming_anomaly_detection/DARA/{model_category}/confusion_matrix_heatmap_{label_type}.pdf", dpi=300)


# model_category = "ngram/results"
# model_category = "CNN/results"
model_category = "transformer/results/"

model_category += 'longformer_continued_cross-entropy'
# model_category += 'longformer_contrastive'
# model_category += 'longformer_cross-entropy'
# model_category += 'roberta_continued_cross-entropy'
# model_category += 'roberta_contrastive'
# model_category += 'roberta_cross-entropy'

# label_type = "model_type"
label_type = "task"
# label_type = "arch" 

# Open the file and read the data
with open(f'Naming_anomaly_detection/data_files/json_files/{label_type}_index_to_label.json', 'r') as f:
    index_to_label = json.load(f)

with open(f'Naming_anomaly_detection/data_files/json_files/{label_type}_label_type.json', 'r') as f:
    label_type = json.load(f)
    
with open(f'Naming_anomaly_detection/data_files/json_files/{label_type}_label_to_domain.json', 'r') as f:
    label_to_domain = json.load(f)

# Load the predictions and targets
all_fold_preds = np.load(f'Naming_anomaly_detection/DARA/{model_category}/{label_type}_all_fold_preds_final.npy')
all_fold_targets = np.load(f'Naming_anomaly_detection/DARA/{model_category}/{label_type}_all_fold_targets_final.npy')

plot_confusion_matrix_heatmap(all_fold_preds, all_fold_targets, index_to_label, label_type, label_to_domain)