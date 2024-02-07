import os
import json
import pickle
import numpy as np
import pandas as pd
import openai
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Set your OpenAI API key here
openai.api_key = os.environ.get("OPENAI_API_KEY")

def load_data(filename):
    """Load data from a JSON file and return a DataFrame."""
    with open(filename) as f:
        data = json.load(f)
    df = pd.DataFrame(list(data.items()), columns=['ModelName', 'Category'])
    return df

def get_embeddings(texts):
    """Get embeddings for a list of texts using OpenAI's API."""
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-large"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

def train_classifier(X_train, X_test, y_train, y_test):
    """Train a text classifier and return the trained model."""
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    predicted = model.predict(X_test)
    print(classification_report(y_test, predicted))

    return model

def main():
    # Load data
    filename = 'cls_data.json'  # Update this path
    df = load_data(filename)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(df['ModelName'], df['Category'], test_size=0.2, random_state=42)

    # Get embeddings for the training and testing data
    X_train_embeddings = get_embeddings(X_train)
    X_test_embeddings = get_embeddings(X_test)

    # Train the classifier
    model = train_classifier(X_train_embeddings, X_test_embeddings, y_train, y_test)

    # Save the model
    with open('name_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    

if __name__ == '__main__':
    main()
