""" This script is used to test the clustering pipeline on a single architecture. """

import json
import os
from vector.ClusterPipeline import ClusterPipeline
from vector.ClusterTools import get_embedding

if __name__ == "__main__":
    assert os.path.exists("data_files/json_files/ground_truth_cls_dataset.json")
    with open("data_files/json_files/ground_truth_cls_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    selected_arch = data[43]
    print(data[43][0].keys())
    cp = ClusterPipeline()
    result = cp.cluster_single_arch_from_dict((selected_arch[0], selected_arch[1], selected_arch[2]), eps=0.3, merge_outlier=True)
    print(result)
    for arch, groups in result.items():
        for group, models in groups.items():
            print(models[0])
            print(get_embedding(models[0]))
            break