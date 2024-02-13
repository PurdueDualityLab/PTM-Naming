""" This script is used to test the clustering pipeline on a single architecture. """

import json
import os
from vector.ClusterPipeline import ClusterPipeline
from vector.ClusterTools import get_embedding, get_silhouette_score_from_cluster

if __name__ == "__main__":
    assert os.path.exists("data_files/json_files/ground_truth_cls_dataset.json")
    with open("data_files/json_files/ground_truth_cls_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_arch = data[13]
    print(data[13][0].keys())
    cp = ClusterPipeline()
    for eps in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        result = cp.cluster_single_arch_from_dict((selected_arch[0], selected_arch[1], selected_arch[2]), eps=eps, merge_outlier=True)
        #print(result)
        silhouette_score = get_silhouette_score_from_cluster(result)
        if silhouette_score is not None:
            silhouette_score = round(silhouette_score, 2)
        else:
            silhouette_score = None
        print(f"EPS: {eps}, Silhouette Score: {silhouette_score}")
    
    
