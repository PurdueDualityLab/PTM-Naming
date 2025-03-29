import random
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, hamming_loss, accuracy_score
import seaborn as sns

import matplotlib.pyplot as plt

from modeling import DARA_classifier
from dataloader import DARA_dataset

from loguru import logger  # Ensure logger is imported if used for logging


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# torch.random.manual_seed(42)  # Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate(model, train_loader, eval_loader, criterion, optimizer, scheduler, device="cuda", epochs=10, eval_interval=10, multi_label=False):
    model.train()  # Set model to training mode
    train_losses = []
    eval_accuracies = []
    all_preds = []
    all_targets = []

    for epoch in range(epochs):
        total_loss = 0
        for data, target, _ in train_loader:  # Assuming data loaders yield (data, target) tuples
            data, target = data.to(device), target.to(device)  # Ensure target is the correct type for loss calculation
            optimizer.zero_grad()  # Clear gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters
            total_loss += loss.item()  # Accumulate the loss

        # Calculate and log the average loss for the epoch
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        logger.info(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

        # Evaluation phase
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            model.eval()  # Set model to evaluation mode
            accuracy, preds, targets = eval(model, eval_loader, device, multi_label)  # Evaluate model
            eval_accuracies.append(accuracy)
            all_preds.extend(preds)  # Collect all predictions
            all_targets.extend(targets)  # Collect all true labels
            logger.info(f'Epoch {epoch+1}, Eval Accuracy: {accuracy:.2f}%')
            model.train()  # Set model back to training mode

        # Step the learning rate scheduler
        scheduler.step()

    return train_losses, eval_accuracies, all_preds, all_targets



# def eval(model, eval_loader, device="cuda"):
#     model.eval()  # Set the model to evaluation mode
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():  # No need to track gradients for evaluation
#         for data, target, _ in eval_loader:  # Change this line to match the expected output
#             data, target = data.to(device), target.to(device).long()
#             output = model(data)
#             _, predicted = torch.max(output, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_targets.extend(target.cpu().numpy())
#     correct = (np.array(all_preds) == np.array(all_targets)).sum()
#     total = len(all_targets)
#     accuracy = 100 * correct / total
#     return accuracy, all_preds, all_targets

def top_2_accuracy(y_true, y_logits):
    top_2 = np.argsort(y_logits, axis=1)[:, -2:]  # Get indices of top 2 predictions
    accuracy = 0
    
    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]  # Get indices of true labels
        for label in true_labels:
            if label in top_2[i]:
                accuracy += 1
                break  # Count a sample as correct if any true label is in top 2
    return accuracy / len(y_true)

def eval(model, eval_loader, device="cuda", multi_label=False):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target, _ in eval_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if multi_label:
                preds = torch.sigmoid(output)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            else:
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    if multi_label:
        # hamming = hamming_loss(all_targets, np.array(all_preds) > 0.5)
        # accuracy = accuracy_score(all_targets, np.array(all_preds) > 0.5)
        # return accuracy * 100, all_preds, all_targets
        accuracy = top_2_accuracy(all_targets, np.array(all_preds)) #calculate top 2 accuracy.
        return accuracy * 100, all_preds, all_targets
    else:
        correct = (np.array(all_preds) == np.array(all_targets)).sum()
        total = len(all_targets)
        accuracy = 100 * correct / total
        return accuracy, all_preds, all_targets



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



def plot_confusion_matrix(preds, targets, index_to_label, label_type):
    with open(f'./Naming_anomaly_detection/data_files/json_files/{label_type}_label_to_domain.json', 'r') as f:
        label_to_domain = json.load(f)
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
        # label_to_domain = {domain: [label for label in labels if label in label_to_index] for domain, labels in label_to_domain.items()}
            
    # labels_ordered = [index_to_label[str(i)] for i in range(len(index_to_label))]
    # Order labels based on domain
    labels_ordered = []
    for domain, models in label_to_domain.items():
        labels_ordered.extend(models)
    # Ensure all labels are included
    labels_ordered = [label for label in labels_ordered if label in index_to_label.values()]
    # # Create a mapping from label to index
    # label_to_index = {label: index for index, label in index_to_label.items()}
    # print(label_to_index)
    label_to_domain = {domain: [label for label in labels if label in labels_ordered] for domain, labels in label_to_domain.items()}
    # print(labels_ordered)
    
    targets = [index_to_label[target] for target in targets]
    preds = [index_to_label[pred] for pred in preds]

    # Calculate the confusion matrix 
    cm = confusion_matrix(targets, preds, labels=labels_ordered)

    # Normalize the confusion matrix by row (i.e., by the number of samples in each true class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
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
    for domain, models in label_to_domain.items():
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
    plt.savefig(f"confusion_matrix_percentage_{label_type}_final.pdf", dpi=300)
    # cm = confusion_matrix(targets, preds)
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convert counts to percentages
    
    # labels = [index_to_label[i] for i in range(len(index_to_label))]
    
    # fig, ax = plt.subplots(figsize=(15, 15))  # Increased figure size
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=labels)
    # disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical', values_format='.1f')  # Rotate x-axis labels, format percentages
    
    # plt.xticks(fontsize=9)  # Adjust fontsize as needed
    # plt.yticks(fontsize=9)  # Adjust fontsize as needed
    # plt.title(f'Confusion Matrix for {label_type}', fontsize=12)
    # plt.xlabel('Predicted label', fontsize=10)
    # plt.ylabel('True label', fontsize=10)
    # colorbar = ax.images[0].colorbar
    # colorbar.set_label('% of True Labels', fontsize=10)
    # colorbar.ax.yaxis.set_label_position('left')
    
    # plt.tight_layout()  # Adjust layout to not cut off labels


def plot_loss(train_losses, epochs, lr, batch_size, label_type):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss over Epochs | LR: {lr}, Batch Size: {batch_size}')
    plt.legend()
    plt.savefig(f"{label_type}_train_loss_{epochs}epochs_lr{lr}_batch{batch_size}_final.png")


def plot_accuracy(test_accuracies, epochs, lr, batch_size, label_type):
    eval_interval = epochs // len(test_accuracies) if len(test_accuracies) > 0 else 1
    x_vals = list(range(eval_interval, epochs + 1, eval_interval))
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, test_accuracies, '-o', label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy over Epochs | LR: {lr}, Eval Batch Size: {batch_size}')
    plt.xticks(x_vals)
    plt.legend()
    plt.savefig(f"{label_type}_test_accuracy_{epochs}epochs_lr{lr}_batch{batch_size}_final.png")


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
    model = DARA_classifier(input_size=input_shape[0], num_classes=num_classes).to(device)
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

    # label_type = "model_type"
    label_type = "task"
    # label_type = "arch" # TODO: This needs a different hyperparameter setting
    ############################
    if label_type == "task":
        vec_path = 'data_cleaned.json'
        multi_label = True
    else:
        vec_path = 'data.json'
        multi_label = False
    # data_loader = DataLoader(vec_path)

    full_dataset = DARA_dataset(dict_path=vec_path, label_type=label_type, multi_label=multi_label)


    index_to_label = full_dataset.get_label_mapping()
    num_classes = full_dataset.get_num_classes()
    input_shape = full_dataset.get_data_shape()
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Input shape: {input_shape}")
    logger.info(f"label type: {label_type}")



    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Variables to store cumulative results
    cumulative_train_losses = []
    cumulative_eval_accuracies = []

    fold = 0  # Counter for current fold

    all_fold_preds = []
    all_fold_targets = []

    for train_index, eval_index in kf.split(full_dataset):
        fold += 1
        logger.info(f"Starting fold {fold}")

        # Create datasets for the current fold
        train_subset = torch.utils.data.Subset(full_dataset, train_index)
        eval_subset = torch.utils.data.Subset(full_dataset, eval_index)

        # Initialize DataLoaders for the current fold
        train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
        eval_loader = DataLoader(eval_subset, batch_size=eval_batch_size, num_workers=0, pin_memory=True)

        # Initialize the model for the current fold
        model = DARA_classifier(input_size=input_shape[1], output_size=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        if multi_label:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Train and evaluate the model on the current fold
        train_losses, eval_accuracies, preds, targets = train_and_evaluate(model, train_loader, eval_loader, criterion, optimizer, scheduler, device=device, epochs=epochs, eval_interval=10, multi_label=multi_label)
        logger.success(f"Fold {fold}, Final Eval Accuracy: {eval_accuracies[-1]:.2f}%")

        all_fold_preds.extend(preds)
        all_fold_targets.extend(targets)
        # Append results from the current fold
        cumulative_train_losses.append(train_losses)
        cumulative_eval_accuracies.append(eval_accuracies)

        # Optionally, save model and results per fold
        '''
        torch.save(model.state_dict(), f'fold_{fold}_model_state_dict_final.pt')
        evaluate_and_save_results(model, eval_loader, index_to_label, device=device, filepath=f'fold_{fold}_{label_type}_evaluation_results_final.json')
        '''

    # After all folds are completed, calculate and log the average performance across all folds
    average_train_loss = [sum(losses) / len(losses) for losses in zip(*cumulative_train_losses)]
    average_eval_accuracy = [sum(accs) / len(accs) for accs in zip(*cumulative_eval_accuracies)]
    logger.info(f"Average Eval Accuracy across all folds: {average_eval_accuracy[-1]:.2f}%")

    # Call plotting functions for the averages
    # plot_loss(average_train_loss, epochs, lr, train_batch_size, label_type)
    # plot_accuracy(average_eval_accuracy, epochs, lr, eval_batch_size, label_type)

    logger.success("5-Fold Cross Validation completed")
    '''
    # Save the inputs to confusion matrix
    np.save(f'{label_type}_all_fold_preds_final.npy', all_fold_preds)
    np.save(f'{label_type}_all_fold_targets_final.npy', all_fold_targets)

    # Save index_to_label and label_type
    with open(f'{label_type}_index_to_label_final.json', 'w') as f:
        json.dump(index_to_label, f)
    
    with open(f'{label_type}_label_type_final.json', 'w') as f:
        json.dump(label_type, f)

    # # Plotting the confusion matrix for all folds
    plot_confusion_matrix(all_fold_preds, all_fold_targets, index_to_label, label_type)
    '''
if __name__ == "__main__":
    # Set random seed
    # torch.manual_seed(0)
    # np.random.seed(0)
    set_seed(0)
    # run_PTMTorrent()
    # run()
    CV_run()


