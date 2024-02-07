import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

def train_and_evaluate(X_train, y_train, X_test, y_test, max_iter):
    """Train a model and evaluate its accuracy on both training and test sets."""
    model = LogisticRegression(max_iter=max_iter, solver='lbfgs', verbose=0)
    model.fit(X_train, y_train)
    
    # Predict and calculate accuracy on the training set
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    # Predict and calculate accuracy on the test set
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    return train_accuracy, test_accuracy, model

if __name__ == '__main__':
    # Load data
    embeddings = np.load('embeddings.npy')
    labels = np.load('labels.npy', allow_pickle=True)

    # Shuffle and split the data
    embeddings, labels = shuffle(embeddings, labels, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    
    iteration_steps = range(0, 100, 5)  # Adjust the step and range as needed
    train_accuracies = []
    test_accuracies = []

    for iters in iteration_steps:
        train_acc, test_acc, model = train_and_evaluate(X_train, y_train, X_test, y_test, iters)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Iterations: {iters}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

    # Plotting the accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_steps, train_accuracies, label='Train Accuracy')
    plt.plot(iteration_steps, test_accuracies, label='Test Accuracy', linestyle='--')
    plt.title('Accuracy vs. Number of Iterations')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_vs_iterations.png")


    with open(f'model.pkl', 'wb') as f:
        pickle.dump(model, f)