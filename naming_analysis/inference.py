# inference.py

import pickle
import numpy as np
import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_embeddings(texts):
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return np.array([item['embedding'] for item in response['data']])

if __name__ == '__main__':
    iters = 20
    # Load the trained model
    with open(f'model_{iters}.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Example text to classify
    example_texts = ['bert-dataset-imagenet-uncased']
    example_embeddings = get_embeddings(example_texts)
    
    print(f"Example prediction of {example_texts}:", model.predict(example_embeddings))