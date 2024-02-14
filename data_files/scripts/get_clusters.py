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

    for model_list_ in selected_arch[0].values():
        model_list = list(model_list_)

    vec_tuple = (selected_arch[0], selected_arch[1], selected_arch[2])

    gsp = GridSearchPipeline(
        model_list = model_list
    )
    eps_list = np.linspace(0.001, 0.2, 200)
    result = gsp.grid_search(
        vec_tuple,
        gsp.get_silhouette_score,
        eps_list,
        4
    )
    print(result)

