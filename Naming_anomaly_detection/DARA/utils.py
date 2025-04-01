import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from matplotlib import pyplot as plt

def top_k_predictions(y_pred_scores, top_k=2):
    y_pred_topk = []
    for i in range(len(y_pred_scores)):
        top_k_indices = np.argsort(y_pred_scores[i])[-top_k:]
        temp_row = np.zeros_like(y_pred_scores[i], dtype=int)
        temp_row[top_k_indices] = 1
        y_pred_topk.append(temp_row)
    return np.array(y_pred_topk)

def top_k_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for i in range(total):
        true_labels = set(np.where(y_true[i] == 1)[0])
        predicted_labels = set(np.where(y_pred[i] == 1)[0])
        if true_labels.intersection(predicted_labels):
            correct += 1
    return correct / total

def eval_metrics(y_true, y_pred, top_k=None):
    if top_k:
        accuracy_fn = top_k_accuracy
    else:
        accuracy_fn = accuracy_score
    results = {}
    results["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    results["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["accuracy"] = accuracy_fn(y_true, y_pred)
    results["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    results["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    results["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    return results

def plot_loss(train_losses, eval_losses, epochs, lr, batch_size, label_type, eval_interval, root_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(eval_interval, epochs + 1, eval_interval), eval_losses, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss over Epochs | LR: {lr}, Batch Size: {batch_size}')
    plt.legend()
    plt.savefig(f"{root_dir}/results/{label_type}_train_loss_{epochs}epochs_lr{lr}_batch{batch_size}_final.png")

def plot_accuracy(test_accuracies, epochs, lr, batch_size, label_type, root_dir):
    eval_interval = epochs // len(test_accuracies) if len(test_accuracies) > 0 else 1
    x_vals = list(range(eval_interval, epochs + 1, eval_interval))
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, test_accuracies, '-o', label='Test Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Test Accuracy over Epochs | LR: {lr}, Eval Batch Size: {batch_size}')
    plt.xticks(x_vals)
    plt.legend()
    plt.savefig(f"{root_dir}/results/{label_type}_test_accuracy_{epochs}epochs_lr{lr}_batch{batch_size}_final.png")