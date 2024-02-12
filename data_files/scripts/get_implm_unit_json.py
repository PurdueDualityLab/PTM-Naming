
import json
import os

if __name__ == "__main__":
    assert os.path.exists("data_files/json_files/ground_truth_cls_data.json")
    with open("data_files/json_files/ground_truth_cls_data.json", "r") as f:
        ground_truth_cls_data = json.load(f)
    implm_unit_list = []
    for name, cls_type in ground_truth_cls_data.items():
        if cls_type == "implementation unit":
            implm_unit_list.append(name)
    with open("data_files/json_files/implm_unit_list.json", "w") as f:
        json.dump(implm_unit_list, f)