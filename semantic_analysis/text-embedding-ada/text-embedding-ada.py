import os
import json
from openai import OpenAI
from scipy.spatial.distance import cosine

from embeddings_utils import get_embedding, cosine_similarity

client = OpenAI()



import prompt

from loguru import logger

OpenAI.api_key = os.environ.get("OPENAI_API_KEY")


prompt = prompt.BACKGROUND



def get_name_embeddings(models, save_results=False):
      
  embedding_model = "text-embedding-3-large"
  embedding_encoding = "cl100k_base"
  max_tokens = 8000 


  cnt = 0
  embeddings = {}
  for arch in models:
    for model in models[arch]:
        if len(model.split("/")) > 1:
            model = model.split("/")[-1]
        logger.info(f"Creating embedding for model {model}")
        embedding = get_embedding(model, embedding_model)
        embeddings[model] = embedding
    break


  if save_results == True:
    # Save embeddings to file
    with open("embeddings.json", "w") as f:
        json.dump(embeddings, f)
  return embeddings


def DBSCAN_clustering(embeddings, threshold=0.5):
    from sklearn.cluster import DBSCAN
    import numpy as np
    X = np.array([embeddings[model] for model in embeddings])
    db = DBSCAN(eps=threshold, min_samples=2).fit(X)
    return db.labels_

with open("../filtered_models.json", "r") as f:
    models = json.load(f)

if not os.path.exists("embeddings.json"):
    embeddings = get_name_embeddings(models, save_results=True)
else:
    with open("embeddings.json", "r") as f:
        embeddings = json.load(f)


clusters = DBSCAN_clustering(embeddings, threshold=0.3)
logger.success(f"Clusters: {clusters}")


# Organize models into clusters
clustered_models = {}
for model, cluster_label in zip(embeddings.keys(), clusters):
    # Convert cluster_label to Python's native int type
    cluster_label = int(cluster_label)  # This line fixes the issue
    if cluster_label not in clustered_models:
        clustered_models[cluster_label] = []
    clustered_models[cluster_label].append(model)

# Save the clusters to a JSON file
clusters_file_path = "name_clusters.json"
with open(clusters_file_path, "w") as f:
    json.dump(clustered_models, f, indent=4)

logger.success(f"Clustered models saved to {clusters_file_path}")