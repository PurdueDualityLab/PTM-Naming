import os
import json
from loguru import logger
from vector.cluster_dataset import ClusterDataset
from vector.ann_vector import ANNVectorTripletArchGroup

if __name__ == "__main__":
    assert os.path.exists("model_collection/modelArch_list.json")
    assert os.path.exists("data_files/json_files/implm_unit_list.json")

    cls_ds = ClusterDataset()

    with open("model_collection/modelArch_list.json", "r") as f:
        model_arch_list = json.load(f)
    with open("data_files/json_files/implm_unit_list.json", "r") as f:
        implm_unit_list = json.load(f)

    triplet_arch_group_list = []
    for arch_name in model_arch_list:
        curr_triplet_group = ANNVectorTripletArchGroup.from_dataset(cls_ds, arch_name)
        if curr_triplet_group is None:
            logger.warning(f"Skipping {arch_name}. Could not find any models.")
            continue
        logger.info(f"Processing {arch_name}. Total models: {len(curr_triplet_group.vector_triplet_list)}")
        remove_count = 0
        for ann_triplet in curr_triplet_group.vector_triplet_list:
            if ann_triplet.model_name not in implm_unit_list:
                curr_triplet_group.remove(ann_triplet.model_name)
                remove_count += 1
        triplet_arch_group_list.append(list(curr_triplet_group.to_dict()))
        logger.info(f"Removed {remove_count} models from {arch_name}. Total models: {len(curr_triplet_group.vector_triplet_list)}")
    with open("data_files/json_files/ground_truth_cls_dataset.json", "w") as f:
        json.dump(triplet_arch_group_list, f)

