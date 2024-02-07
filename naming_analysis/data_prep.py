# data_preparation.py

import json
import numpy as np
import pandas as pd
import openai
from tqdm.auto import tqdm
from loguru import logger
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")

def load_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return pd.DataFrame(list(data.items()), columns=['ModelName', 'Category'])

def get_embeddings(texts, batch_size=20):
    all_embeddings = []
    logger.info(f"Creating embeddings for {len(texts)} texts")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Batch Progress"):
        batch_texts = texts[i:i+batch_size]
        try:
            response = openai.Embedding.create(
                input=[text.split('/')[-1] for text in batch_texts],
                model="text-embedding-3-small"
            )
            batch_embeddings = [item['embedding'] for item in response['data']]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
    return np.array(all_embeddings)

if __name__ == '__main__':
    filename = 'cls_data.json'  # Path to your JSON file
    df = load_data(filename)
    embeddings = get_embeddings(df['ModelName'].tolist())
    
    # Save embeddings and labels for later use
    np.save('embeddings.npy', embeddings)
    df['Category'].to_numpy().dump('labels.npy')
