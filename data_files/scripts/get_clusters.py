import json
import os
from vector.ClusterPipeline import ClusterPipeline

if __name__ == "__main__":
    assert os.path.exists("data_files/json_files/ground_truth_cls_dataset.json")
    with open("data_files/json_files/groundtruth_cls_dataset.json", "r") as f:
        data = json.load(f)
    selected_arch = data["Wav2Vec2ForCTC"]
    cp = ClusterPipeline()
    result, outlier = cp.cluster_single_arch_from_dict((selected_arch[0], selected_arch[1], selected_arch[2]), eps=0.3)
    print(result, outlier)