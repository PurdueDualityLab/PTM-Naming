"""
This module contains the classes for the APTMVector, APTMVectorTriplet, 
and APTMVectorTripletArchGroup.
"""
from typing import List
from vector.cluster_dataset import ClusterDataset
from APTM.abstract_neural_network import AbstractNN

class APTMVector():
    """
    This class contains the vector representation of the model.

    Attributes:
        model_name (str): The name of the model.
        content (dict): The vector representation of the model.
        category (str): The category of the vector representation.
    """
    def __init__(self, model_name: str, content: dict, category: str):
        self.model_name = model_name
        self.content = content
        self.category = category

    @staticmethod
    def from_dict(model_name, category, vec_dict):
        """
        This function creates an APTMVector from a dictionary.

        Args:
            model_name (str): The name of the model.
            category (str): The category of the vector representation.
            vec_dict (dict): The vector representation of the model.

        Returns:
            APTMVector: The APTMVector object.
        """
        return APTMVector(model_name, vec_dict, category)

    def __repr__(self) -> str:
        return str(self.content)

class APTMVectorTriplet():
    """
    This class contains the vector representation of the model.

    Attributes:
        model_name (str): The name of the model.
        vector_l (APTMVector): The vector representation of the model's layers.
        vector_p (APTMVector): The vector representation of the model's parameters.
        vector_d (APTMVector): The vector representation of the model's dimensions.
    """
    def __init__(
        self,
        model_name: str,
        vector_l: APTMVector,
        vector_p: APTMVector,
        vector_d: APTMVector
    ):
        self.model_name = model_name
        self.vector_l = vector_l
        self.vector_p = vector_p
        self.vector_d = vector_d

    @staticmethod
    def from_aptm(model_name: str, aptm: AbstractNN, *args, **kwargs):
        """
        This function creates an APTMVectorTriplet from an AbstractNN.

        Args:
            model_name (str): The name of the model.
            aptm (AbstractNN): The AbstractNN object.

        Returns:
            APTMVectorTriplet: The APTMVectorTriplet object.
        """
        fv_l, fv_p, fv_d = aptm.vectorize(*args, **kwargs)
        triplet = APTMVectorTriplet(
            model_name,
            APTMVector.from_dict(model_name, "l", fv_l),
            APTMVector.from_dict(model_name, "p", fv_p),
            APTMVector.from_dict(model_name, "d", fv_d),
        )
        return triplet

    @staticmethod
    def from_dict(model_name, vec_l, vec_p, vec_d):
        """
        This function creates an APTMVectorTriplet from a dictionary.

        Args:
            model_name (str): The name of the model.
            vec_l (dict): The vector representation of the model's layers.
            vec_p (dict): The vector representation of the model's parameters.
            vec_d (dict): The vector representation of the model's dimensions.

        Returns:
            APTMVectorTriplet: The APTMVectorTriplet object.
        """
        return APTMVectorTriplet(
            model_name,
            APTMVector.from_dict(model_name, "l", vec_l),
            APTMVector.from_dict(model_name, "p", vec_p),
            APTMVector.from_dict(model_name, "d", vec_d)
        )

    def to_dict(self):
        """
        This function returns the dictionary representation of the APTMVectorTriplet.

        Returns:
            dict: The dictionary representation of the APTMVectorTriplet.
        """
        return self.vector_l, self.vector_p, self.vector_p

    def __repr__(self) -> str:
        return str(
            {"model_name": self.model_name,
             "content": {"l": self.vector_l, "p": self.vector_p, "d": self.vector_d}
            }
        )

class APTMVectorTripletArchGroup():
    """
    This class contains the vector representation of the model.

    Attributes:
        arch_name (str): The name of the architecture.
        vector_triplet_list (List[APTMVectorTriplet]): The list of APTMVectorTriplets.
    """
    def __init__(
        self,
        arch_name: str,
        vector_triplet_list: List[APTMVectorTriplet]
    ):
        self.arch_name = arch_name
        self.vector_triplet_list = vector_triplet_list

    @staticmethod
    def from_dataset(dataset: ClusterDataset, arch_name: str):
        """
        This function creates an APTMVectorTripletArchGroup from a ClusterDataset.

        Args:
            dataset (ClusterDataset): The ClusterDataset object.
            arch_name (str): The name of the architecture.
        
        Returns:
            APTMVectorTripletArchGroup: The APTMVectorTripletArchGroup object.
        """
        if arch_name not in dataset.vec_l:
            return None

        raw_dict_l = dataset.get("l", arch_name, "arch")[arch_name]
        raw_dict_p = dataset.get("p", arch_name, "arch")[arch_name]
        raw_dict_d = dataset.get("d", arch_name, "arch")[arch_name]

        triplet_list = []
        for model_name in raw_dict_l.keys():
            curr_triplet = APTMVectorTriplet(
                model_name,
                APTMVector.from_dict(model_name, "l", raw_dict_l[model_name]),
                APTMVector.from_dict(model_name, "p", raw_dict_p[model_name]),
                APTMVector.from_dict(model_name, "d", raw_dict_d[model_name]),
            )
            triplet_list.append(curr_triplet)
        return APTMVectorTripletArchGroup(arch_name, triplet_list)

    def add(self, triplet: APTMVectorTriplet):
        """
        This function adds an APTMVectorTriplet to the APTMVectorTripletArchGroup.

        Args:
            triplet (APTMVectorTriplet): The APTMVectorTriplet to be added.
        
        Returns:
            None
        """
        self.vector_triplet_list.append(triplet)

    def get(self, i):
        """
        This function returns the i-th APTMVectorTriplet.

        Args:
            i (int): The index of the APTMVectorTriplet.
        
        Returns:
            APTMVectorTriplet: The i-th APTMVectorTriplet.
        """
        return self.vector_triplet_list[i]

    def size(self):
        """
        This function returns the size of the APTMVectorTripletArchGroup.

        Returns:
            int: The size of the APTMVectorTripletArchGroup.
        """
        return len(self.vector_triplet_list)

    def to_dict(self):
        """
        This function returns the dictionary representation of the APTMVectorTripletArchGroup.

        Returns:
            dict: The dictionary representation of the APTMVectorTripletArchGroup.
        """
        vec_l, vec_p, vec_d = \
        {self.arch_name: {}}, \
        {self.arch_name: {}}, \
        {self.arch_name: {}}
        for vector_triplet in self.vector_triplet_list:
            model_name = vector_triplet.model_name
            vec_l[self.arch_name][model_name] = vector_triplet.vector_l.content
            vec_p[self.arch_name][model_name] = vector_triplet.vector_p.content
            vec_d[self.arch_name][model_name] = vector_triplet.vector_d.content
        return vec_l, vec_p, vec_d

    @staticmethod
    def from_dict(vec_l, vec_p, vec_d):
        """
        This function creates an APTMVectorTripletArchGroup from a dictionary.

        Args:
            vec_l (dict): The vector representation of the model's layers.
            vec_p (dict): The vector representation of the model's parameters.
            vec_d (dict): The vector representation of the model's dimensions.
        
        Returns:
            APTMVectorTripletArchGroup: The APTMVectorTripletArchGroup object.
        """
        arch_name = list(vec_l.keys())[0]
        triplet_list = []
        for model_name in vec_l[arch_name].keys():
            curr_triplet = APTMVectorTriplet(
                model_name,
                APTMVector.from_dict(model_name, "l", vec_l[arch_name][model_name]),
                APTMVector.from_dict(model_name, "p", vec_p[arch_name][model_name]),
                APTMVector.from_dict(model_name, "d", vec_d[arch_name][model_name]),
            )
            triplet_list.append(curr_triplet)
        return APTMVectorTripletArchGroup(arch_name, triplet_list)

    def to_array(self):
        """
        This function returns the array representation of the APTMVectorTripletArchGroup.

        Returns:
            dict: The array representation of the APTMVectorTripletArchGroup.
        """
        keys_l, keys_p, keys_d = [], [], []

        for vector_triplet in self.vector_triplet_list:
            keys_l.append(list(vector_triplet.vector_l.content.keys()))
            keys_p.append(list(vector_triplet.vector_p.content.keys()))
            keys_d.append(list(vector_triplet.vector_d.content.keys()))

        def merge_keys(keys_arr):
            merged_keys_arr = []
            for model_specific_keys in keys_arr:
                for key in model_specific_keys:
                    merged_keys_arr.append(key)
            return sorted(list(set(merged_keys_arr)))

        keys_l, keys_p, keys_d = merge_keys(keys_l), merge_keys(keys_p), merge_keys(keys_d)

        vec_l, vec_p, vec_d = \
            {self.arch_name: dict()}, \
            {self.arch_name: dict()}, \
            {self.arch_name: dict()}
        for vector_triplet in self.vector_triplet_list:
            curr_vecarr_l, curr_vecarr_p, curr_vecarr_d = \
                [0 for i in range(len(keys_l))], \
                [0 for i in range(len(keys_p))], \
                [0 for i in range(len(keys_d))]
            def fill_pure_int_vec(keys_arr, vec_arr, vec_triplet_content):
                for i, curr_key in enumerate(keys_arr):
                    if curr_key in vec_triplet_content:
                        vec_arr[i] += vec_triplet_content[curr_key]
            fill_pure_int_vec(keys_l, curr_vecarr_l, vector_triplet.vector_l.content)
            fill_pure_int_vec(keys_p, curr_vecarr_p, vector_triplet.vector_p.content)
            fill_pure_int_vec(keys_d, curr_vecarr_d, vector_triplet.vector_d.content)
            vec_l[self.arch_name][vector_triplet.model_name] = curr_vecarr_l
            vec_p[self.arch_name][vector_triplet.model_name] = curr_vecarr_p
            vec_d[self.arch_name][vector_triplet.model_name] = curr_vecarr_d

        return vec_l, vec_p, vec_d

    def remove(self, item_name) -> None:
        """
        This function removes an APTMVectorTriplet from the APTMVectorTripletArchGroup.

        Args:
            item_name (str): The name of the APTMVectorTriplet to be removed.
        
        Returns:
            None
        """
        self.vector_triplet_list = [
            triplet for triplet in self.vector_triplet_list
            if triplet.model_name != item_name
        ]

    def __repr__(self) -> str:
        return str(self.vector_triplet_list)

if __name__ == "__main__":
    # ds = ClusterDataset()
    # arch_group = APTMVectorTripletArchGroup.from_dataset(ds, "WavLMForCTC")
    # assert arch_group is not None
    # d_arch_group = arch_group.to_dict()
    # print(arch_group)
    # arch_group_recnstr = APTMVectorTripletArchGroup.from_dict(*d_arch_group)
    # print(arch_group_recnstr)
    # Test ngram
    aptm = AbstractNN.from_huggingface("microsoft/resnet-18")
    aptm.export_aptm("rn18_test.json")
    aptm_triplet = APTMVectorTriplet.from_aptm("rn18", aptm, ngram = 3)
    print(aptm_triplet.vector_l.content)
