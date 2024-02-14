""" This script is used to test the clustering pipeline on a single architecture. """

import json
import os
import numpy as np
from vector.ClusterPipeline import ClusterPipeline
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

    gsp = GridSearchPipeline(
        model_list = model_list
    )
    # eps_list = np.logspace(-4, 4, 50)
    # result = gsp.grid_search(
    #     vec_tuple,
    #     gsp.get_silhouette_score,
    #     eps_list,
    #     10
    # )
    # print(result)

    result = gsp.search_optimal_eps(
        vec_tuple,
        gsp.get_silhouette_score
    )

    print(result)

    # result = ClusterPipeline().cluster_single_arch_from_dict(
    #     vec_tuple,
    #     eps=0.001,
    #     merge_outlier=True
    # )
    # print(result)

