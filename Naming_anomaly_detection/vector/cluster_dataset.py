"""
This module is used to load the cluster dataset and return the vector
"""
from typing import Dict
import os
import pickle
import dotenv

class ClusterDataset():
    """
    This class is used to load the cluster dataset and return the vector
    representation of the model.

    Attributes:
        cluster_dir: The directory where the cluster dataset is stored
        vec_l: The vector representation of the layers
        vec_p: The vector representation of the parameters
        vec_d: The vector representation of the dimensions
        key_l: The key of the vector representation of the layers
        key_p: The key of the vector representation of the parameters
        key_d: The key of the vector representation of the dimensions
    """
    def __init__(self, cluster_dir=None):
        if cluster_dir is None:
            dotenv.load_dotenv(".env", override=True)
            cluster_dir = os.getenv("CLUSTER_DIR")
        if cluster_dir is None:
            raise ValueError("Cluster directory not found.")
        self.cluster_dir = cluster_dir
        self.vec_l = self.load_pkl(self.cluster_dir + "/vec_l.pkl")
        self.vec_p = self.load_pkl(self.cluster_dir + "/vec_p.pkl")
        self.vec_d = self.load_pkl(self.cluster_dir + "/vec_d.pkl")
        self.key_l = self.load_pkl(self.cluster_dir + "/k_l.pkl")
        self.key_p = self.load_pkl(self.cluster_dir + "/k_p.pkl")
        self.key_d = self.load_pkl(self.cluster_dir + "/k_d.pkl")

    def load_pkl(self, file_loc):
        """
        This function loads a pickle file and returns the content.

        Args:
            file_loc: The location of the pickle file

        Returns:
            The content of the pickle file
        """
        with open(file_loc, 'rb') as f:
            return pickle.load(f)

    def get(
        self,
        vec_category: str,
        item_name: str,
        mode: str = "arch"
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        This function returns the vector representation of the model.

        Args:
            vec_category: The category of the vector. Choose from l, p, or d.
            item_name: The name of the model
            mode: The mode of the vector representation. Choose from arch or param.

        Returns:
            The vector representation of the model
        """
        if vec_category == "l":
            vec, key = self.vec_l, self.key_l
        elif vec_category == "p":
            vec, key = self.vec_p, self.key_p
        elif vec_category == "d":
            vec, key = self.vec_d, self.key_d
        else:
            raise ValueError("Invalid name of vector. Choose from l, p, or d.")

        if mode == "arch":
            if item_name not in vec:
                raise ValueError(f"Model architecture {item_name} not found.")
            model_vec_list = vec[item_name]
            recnstr_dict = {item_name: dict()}
            for model_name, model_vec in model_vec_list.items():
                recnstr_dict[item_name][model_name] = dict()
                for i, dict_key in enumerate(key):
                    dict_val = model_vec[i]
                    if dict_val == 0:
                        continue
                    recnstr_dict[item_name][model_name][dict_key] = dict_val
            return recnstr_dict
        else:
            raise ValueError(f"Mode {mode} not supported.")


if __name__ == "__main__":
    ds = ClusterDataset()
    v = ds.get("l", "WavLMForCTC")
    print(v)
