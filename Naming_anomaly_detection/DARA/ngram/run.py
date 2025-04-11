import random
import os
import json
import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import seaborn as sns
from collections import defaultdict

import matplotlib.pyplot as plt

from modeling import MLP_classifier
from dataloader import DARA_dataset
from DARA.utils import top_k_predictions, eval_metrics, plot_loss, plot_accuracy

from loguru import logger  # Ensure logger is imported if used for logging


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
num_classes = None
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.random.manual_seed(42)  # Set random seed for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def eval(model, eval_loader, criterion, device="cuda", multi_label=False, top_k=None):
    global num_classes

    model.eval()
    all_preds_scores = []
    all_targets = []
    total_loss = 0
    with torch.no_grad():
        for data, target, _ in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            all_preds_scores.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    eval_loss = total_loss / len(eval_loader)
    logger.info(f'Eval Loss: {eval_loss:.4f}')

    all_preds_scores = np.array(all_preds_scores)
    all_targets = np.array(all_targets)

    if multi_label:
        all_preds = top_k_predictions(all_preds_scores, 1)
        metrics_1 = eval_metrics(all_targets, all_preds, 1)
        preds_2 = top_k_predictions(all_preds_scores, 2)
        metrics_2 = eval_metrics(all_targets, preds_2, 2)
        preds_3 = top_k_predictions(all_preds_scores, 3)
        metrics_3 = eval_metrics(all_targets, preds_3, 3)
        metrics = {**metrics_1, **metrics_2, **metrics_3}
    else:
        all_preds = np.argmax(all_preds_scores, axis=1)
        metrics = eval_metrics(all_targets, all_preds)
    
    
    return eval_loss, metrics, all_preds, all_targets

def train_and_evaluate(model, train_loader, eval_loader, criterion, optimizer, scheduler, device="cuda", epochs=10, eval_interval=10, multi_label=False, top_k=None):
    model.train()  # Set model to training mode
    train_losses, eval_losses = [], []
    eval_results = defaultdict(list)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for data, target, _ in train_loader:  # Assuming data loaders yield (data, target) tuples
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Calculate and log the average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

        # Evaluation phase
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            model.eval()  # Set model to evaluation mode
            eval_loss, metrics, preds, targets = eval(model, eval_loader, criterion, device, multi_label, top_k)  # Evaluate model
            eval_losses.append(eval_loss)
            for k, v in metrics.items():
                eval_results[k].append(v)

            logger.info(f'Epoch {epoch+1}, Eval Metrics: {", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])}')

        # Step the learning rate scheduler
        scheduler.step()

    return train_losses, eval_losses, eval_results, preds, targets

def evaluate_and_save_results(model, eval_loader, index_to_label, device="cuda", filepath='evaluation_results.json'):
    model.eval()  # Set the model to evaluation mode
    incorrect_results = []

    with torch.no_grad():  # No need to track gradients for evaluation
        for data, target, model_names in eval_loader:  # Now expecting a third return value from the loader
            data, target = data.to(device), target.to(device).long()  # Ensure data and labels are on the correct device
            output = model(data)
            _, predicted = torch.max(output, 1)  # Get the index of the max log-probability as the predicted label

            # Identify and record incorrect predictions
            incorrect_indices = (predicted != target).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                incorrect_result = {
                    "original_model_name": model_names[idx],  # Add the original model name
                    "predicted_label": index_to_label[predicted[idx].item()],  # Convert predicted label index to category
                    "ground_truth_label": index_to_label[target[idx].item()]  # Convert ground truth label index to category
                }
                incorrect_results.append(incorrect_result)

    # Save the incorrect results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(incorrect_results, f, indent=4)

    logger.success(f"Saved incorrect predictions to {filepath}")

    print(f"Saved incorrect predictions to {filepath}")

def plot_confusion_matrix(preds, targets, index_to_label, label_type, root_dir, top_k=1):
    with open(f'Naming_anomaly_detection/data_files/json_files/{label_type}_label_to_domain.json', 'r') as f:
        domain_to_label = json.load(f)
    if label_type == "arch":
        # Randomly select up to 30 labels for each domain
        selected_labels_per_domain = {}
        all_possible_labels_in_data = set(index_to_label.values())
        for domain, labels in domain_to_label.items():
            valid_labels_in_domain = sorted(list(set(labels) & all_possible_labels_in_data))
            if valid_labels_in_domain:
                selected_labels_per_domain[domain] = random.sample(valid_labels_in_domain, min(len(valid_labels_in_domain), 30))

        for domain, labels in selected_labels_per_domain.items():
            domain_to_label[domain] = sorted(labels)

        selected_labels_set = {label for labels in selected_labels_per_domain.values() for label in labels}

        # Create new mappings based on selected labels
        label_to_index = {label: idx for idx, label in index_to_label.items() if label in selected_labels_set}
        index_to_label = {idx: label for idx, label in index_to_label.items() if label in selected_labels_set}

        filtered_preds_indices = []
        filtered_targets_indices = []
        indices_to_keep = []

        for i in range(len(preds)):
            pred = preds[i]
            target = targets[i]
            if pred in index_to_label and index_to_label[pred] in selected_labels_set and \
            target in index_to_label and index_to_label[target] in selected_labels_set:
                indices_to_keep.append(i)

        filtered_preds_indices = [preds[i] for i in indices_to_keep]
        filtered_targets_indices = [targets[i] for i in indices_to_keep]

        preds = np.array(filtered_preds_indices)
        targets = np.array(filtered_targets_indices)

    # Order labels based on domain (using the selected labels)
    labels_ordered = []
    for domain, labels in domain_to_label.items():
        labels_ordered.extend(labels)
        
    # Ensure uniqueness and maintain order as much as possible
    # labels_ordered = sorted(list(set(labels_ordered)))
    # print(labels_ordered)

    # Convert predictions and targets (which are still indices) to the selected label names
    final_targets = [index_to_label.get(target) for target in targets]
    final_preds = [index_to_label.get(pred) for pred in preds]

    # Remove None values if any predictions or targets were not in the selected labels
    final_targets = [t for t in final_targets if t is not None]
    final_preds = [p for p in final_preds if p is not None]

    # Calculate the confusion matrix
    cm = confusion_matrix(final_targets, final_preds, labels=labels_ordered)

    # Normalize the confusion matrix by row
    cm_normalized = np.zeros_like(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    non_zero_rows = row_sums > 0
    cm_normalized[non_zero_rows[:, 0]] = cm[non_zero_rows[:, 0]].astype('float') / row_sums[non_zero_rows[:, 0]]
    
    # Normalize the confusion matrix by row, handling zero sums
    # cm_normalized = np.zeros_like(cm, dtype=float)
    # row_sums = cm.sum(axis=1, keepdims=True)
    # non_zero_rows = row_sums > 0
    # # Use boolean indexing along the first dimension (rows)
    # cm_normalized[non_zero_rows[:, 0]] = cm[non_zero_rows[:, 0]].astype('float') / row_sums[non_zero_rows[:, 0]]
    plt.figure(figsize=(10, 7))  # Adjust size as needed
    # Plot the heatmap without annotations but with grid lines
    ax = sns.heatmap(cm_normalized, annot=False, cmap='Blues', xticklabels=labels_ordered, yticklabels=labels_ordered, linewidths=.5)
    
    # Decrease the font size of the category labels for both axes
    if label_type == "task":
        fontsize = 10
    else:
        fontsize = 5
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)  # Adjust fontsize for x-axis labels # model_type: 5, task:10
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)  # Adjust fontsize for y-axis labels
    
    # Remove the tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    if label_type == "task":
        fontsize = 10
    else:
        fontsize = 8
    # Add domain labels below the x-axis
    domain_positions = {}
    current_position = 0
    for domain, models in domain_to_label.items():
        domain_positions[domain] = (current_position, current_position + len(models) - 1)
        current_position += len(models)

    for i, (domain, (start, end)) in enumerate(domain_positions.items()):
        if domain == "Multimodal" and label_type == "task":
            ax.text((start + end+1) / 2, len(labels_ordered)-0.1, domain, ha='center', va='bottom', fontsize=fontsize, color='black', rotation=90) # fontsize - model_type:8, task:10
            ax.text(1.5, (start + end + 1) / 2, domain, ha='center', va='center', fontsize=fontsize, color='black')
        else:
            ax.text((start + end+1) / 2, len(labels_ordered)-0.1, domain, ha='center', va='bottom', fontsize=fontsize, color='black')
            ax.text(0.5, (start + end + 1) / 2, domain, ha='center', va='center', fontsize=fontsize, color='black', rotation=90)
        if i < len(domain_positions) - 1:
            ax.axvline(x=end+1, color='gray', linestyle='--', linewidth=0.5)
            ax.axhline(y=end+1, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if label_type == 'task':
        label_type = f'task_{top_k}'
    plt.savefig(f"{root_dir}/results/confusion_matrix_{label_type}_final.pdf", dpi=300)

# outdated
def run():
    ############################
    # hyperparameters
    epochs = 100
    lr = 1e-5
    train_batch_size = 256
    eval_batch_size = 32

    # label_type = "model_type"
    label_type = "arch"
    # label_type = "task"
    ############################

    # vec_path = './data_cleaned.json'
    vec_path = './data_cleaned_full_arch.json'

    
    
    data_loader = DataLoader(vec_path)

    # Create the full dataset
    full_dataset = DARA_dataset(dict_path=vec_path, label_type=label_type)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(full_dataset, [train_size, eval_size])

    # Print out the length of the datasets
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(eval_dataset)}")

    # Initialize DataLoaders for both training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, num_workers=4, pin_memory=True)

    index_to_label = full_dataset.get_label_mapping()
    num_classes = full_dataset.get_num_classes()
    input_shape = full_dataset.get_data_shape()

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Input shape: {input_shape}")
    model = MLP_classifier(input_size=input_shape[0], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Initialize the StepLR scheduler
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


    # Training and evaluation process
    train_losses, eval_accuracies = train_and_evaluate(model, train_loader, eval_loader, criterion, optimizer, scheduler, epochs=epochs, eval_interval=10)

    # Call plotting functions
    plot_loss(train_losses, epochs, lr, train_batch_size, label_type)
    plot_accuracy(eval_accuracies, epochs, lr, eval_batch_size, label_type)

    evaluate_and_save_results(model, eval_loader, index_to_label, filepath=f'{label_type}_evaluation_results.json')

def CV_run():
    from sklearn.model_selection import KFold
    ############################
    # hyperparameters
    epochs = 50
    lr = 1e-3
    train_batch_size = 256
    eval_batch_size = 32
    eval_interval=10
    # label_type = "model_type" # 50 epochs, lr=1e-3, batch_size=256 macro_recall: 0.97 (± 0.02) macro_precision: 0.98 (± 0.01) macro_f1: 0.97 (± 0.02) accuracy: 0.99 (± 0.01)
    # label_type = "task"
    label_type = "arch" # 50 epochs, lr=1e-3, batch_size=256 macro_recall: 0.62 (± 0.01) macro_precision: 0.58 (± 0.01) macro_f1: 0.58 (± 0.01) accuracy: 0.64 (± 0.01)
    ############################
    top_k = None
    if label_type == "task":
        vec_path = 'Naming_anomaly_detection/DARA/ngram/data/data_cleaned.json'
        multi_label = True
        top_k = 1
        # vec_path = 'data_cleaned_prev.json'
    else:
        vec_path = 'Naming_anomaly_detection/DARA/ngram/data/data.json'
        multi_label = False
        # vec_path = 'data_prev.json'
    # data_loader = DataLoader(vec_path)

    full_dataset = DARA_dataset(dict_path=vec_path, label_type=label_type)
    
    global num_classes
    index_to_label = full_dataset.get_label_mapping()
    num_classes = full_dataset.get_num_classes()
    input_shape = full_dataset.get_data_shape()
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"label type: {label_type}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Variables to store cumulative results
    cumulative_train_losses, cumulative_eval_losses = [], []
    cumulative_eval_metrics = defaultdict(list)

    fold = 0  # Counter for current fold

    all_fold_preds = []
    all_fold_targets = []

    for train_index, eval_index in kf.split(full_dataset):
        fold += 1
        logger.info(f"Starting fold {fold}")

        # Create datasets for the current fold
        train_subset = torch.utils.data.Subset(full_dataset, train_index)
        eval_subset = torch.utils.data.Subset(full_dataset, eval_index)

        g = torch.Generator()
        g.manual_seed(42+fold)

        # Initialize DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, num_workers=0, pin_memory=True, worker_init_fn=seed_worker, generator=g)
        # Initialize the model for the current fold
        model = MLP_classifier(input_size=input_shape[1], output_size=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        if label_type == 'task':
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Train and evaluate the model on the current fold
        train_losses, eval_losses, eval_metrics, preds, targets = train_and_evaluate(model, 
                                                                        train_loader, 
                                                                        eval_loader, 
                                                                        criterion, 
                                                                        optimizer, 
                                                                        scheduler, 
                                                                        device=device, 
                                                                        epochs=epochs, 
                                                                        eval_interval=eval_interval, 
                                                                        multi_label=multi_label,
                                                                        top_k=top_k
                                                                        )
        logger.info(f'Fold {fold}, Final Eval Metrics: {", ".join([f"{k}: {v[-1]:.2f}" for k, v in eval_metrics.items()])}')

        all_fold_preds.extend(preds)
        all_fold_targets.extend(targets)
        # Append results from the current fold
        cumulative_train_losses.append(train_losses)
        cumulative_eval_losses.append(eval_losses)
        for key, value in eval_metrics.items():
            cumulative_eval_metrics[key].append(value)

        # Optionally, save model and results per fold
        '''
        torch.save(model.state_dict(), f'fold_{fold}_model_state_dict_final.pt')
        evaluate_and_save_results(model, eval_loader, index_to_label, device=device, filepath=f'fold_{fold}_{label_type}_evaluation_results_final.json')
        '''

    # After all folds are completed, calculate and log the average performance across all folds
    average_train_loss = [sum(losses) / len(losses) for losses in zip(*cumulative_train_losses)]
    average_eval_loss = [sum(losses) / len(losses) for losses in zip(*cumulative_eval_losses)]
    # average_eval_accuracy = [sum(accs) / len(accs) for accs in zip(*cumulative_eval_metrics['accuracy'])]
    # logger.info(f"Average Eval Accuracy across all folds: {average_eval_accuracy[-1]:.2f}%")

    logger.success("5-Fold Cross Validation completed")
    print(f"top_k: {top_k}")
    for k, v in cumulative_eval_metrics.items():
        last_element = [sublist[-1] for sublist in v]
        mean = np.mean(last_element)
        std = np.std(last_element)
        print(f"{k}: {mean:.4f} (± {std:.4f})")
    # Call plotting functions for the averages
    root_dir = 'Naming_anomaly_detection/DARA/ngram'
    plot_loss(average_train_loss, average_eval_loss, epochs, lr, train_batch_size, label_type, eval_interval, root_dir, top_k)
    # plot_accuracy(average_eval_accuracy, epochs, lr, eval_batch_size, label_type, root_dir, top_k)
        
    '''
    '''
    # Save the inputs to confusion matrix
    np.save(f'{root_dir}/results/{label_type}_all_fold_preds_final.npy', all_fold_preds)
    np.save(f'{root_dir}/results/{label_type}_all_fold_targets_final.npy', all_fold_targets)

    # Save index_to_label and label_type
    with open(f'{root_dir}/results/{label_type}_index_to_label_final.json', 'w') as f:
        json.dump(index_to_label, f)
    
    with open(f'{root_dir}/results/{label_type}_label_type_final.json', 'w') as f:
        json.dump(label_type, f)
    # # Plotting the confusion matrix for all folds
    if label_type != 'task':
        plot_confusion_matrix(all_fold_preds, all_fold_targets, index_to_label, label_type, root_dir, top_k)
    
if __name__ == "__main__":
    # Set random seed
    set_seed(0)
    torch.use_deterministic_algorithms(True)
    CV_run()
