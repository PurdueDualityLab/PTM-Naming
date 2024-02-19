""" This script is used to test the clustering pipeline on a single architecture. """

import json
import os
import numpy as np
from matplotlib import pyplot as plt
from vector.cluster_pipeline import ClusterPipeline
from vector.cluster_tools import GridSearchPipeline

if __name__ == "__main__":
    assert os.path.exists("data_files/json_files/ground_truth_cls_dataset.json")
    with open("data_files/json_files/ground_truth_cls_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_arch = data[13]
    print(data[13][0].keys())

    for model_list_full in selected_arch[0].values():
        model_list = [model_name.split('/')[0] for model_name in model_list_full.keys()]

    vec_tuple = (selected_arch[0], selected_arch[1], selected_arch[2])

    model_vec = ClusterPipeline().get_model_vec_from_dict(vec_tuple)

    def normalize_vec(vec_):
        return vec_ / np.linalg.norm(vec_)
    
    for model_family in model_vec:
        for model_name in model_vec[model_family]:
            model_vec[model_family][model_name] = normalize_vec(model_vec[model_family][model_name])
    
    gsp = GridSearchPipeline(
        model_embeddings = model_vec
    )
    eps_list = np.linspace(0.01, 10, 1000)
    result = gsp.grid_search(
        vec_tuple,
        gsp.combination_metric,
        eps_list,
        10
    )
    result = sorted(result.items(), key=lambda x: x[0], reverse=True)
    for eps, score in result:
        print(f"eps: {round(eps, 4)}, score: {round(score, 3)}")

    # Extract eps and score values
    eps_values, score_values = zip(*result)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(eps_values, score_values, marker='o', linestyle='-', color='b')
    plt.title('Silhouette Score - DBI vs EPS')
    plt.xlabel('EPS')
    plt.ylabel('Silhouette Score - DBI')
    plt.grid(True)

    # Save the plot as an image
    plt.savefig('eps_vs_score.png')
    plt.close()
