import os
from openai import OpenAI
import dotenv
from sklearn.metrics import silhouette_score
import numpy as np


def get_embedding(text):
    dotenv.load_dotenv(".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def get_silhouette_score_from_cluster(result):
    embeddings = []
    labels = []

    for cluster_name, groups in result.items():
        for label, models in groups.items():
            for model in models:
                embedding = get_embedding(model)
                embeddings.append(embedding)
                labels.append(label)

    X = np.array(embeddings)
    y = np.array(labels, dtype=int)

    if len(set(y)) < 2:
        return None  # Or handle this case as you see fit

    silhouette_avg = silhouette_score(X, y)
    # print(f'Silhouette Score: {silhouette_avg:.2f}')
    return silhouette_avg