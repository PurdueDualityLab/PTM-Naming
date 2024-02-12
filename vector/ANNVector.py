
from typing import List
from vector.ClusterDataset import ClusterDataset
from ANN.AbstractNN import AbstractNN

class ANNVector():

    def __init__(self, model_name: str, content: dict, category: str):
        self.model_name = model_name
        self.content = content
        self.category = category

    @staticmethod
    def from_dict(model_name, category, vec_dict):
        return ANNVector(model_name, vec_dict, category)
    
    def __repr__(self) -> str:
        return str(self.content)

class ANNVectorTriplet():

    def __init__(
        self,
        model_name: str,
        vector_l: ANNVector,
        vector_p: ANNVector,
        vector_d: ANNVector
    ):
        self.model_name = model_name
        self.vector_l = vector_l
        self.vector_p = vector_p
        self.vector_d = vector_d
    
    @staticmethod
    def from_ANN(model_name: str, ann: AbstractNN):
        fv_l, fv_p, fv_d = ann.vectorize()
        triplet = ANNVectorTriplet(
            model_name,
            ANNVector.from_dict(model_name, "l", fv_l),
            ANNVector.from_dict(model_name, "p", fv_p),
            ANNVector.from_dict(model_name, "d", fv_d),
        )
        return triplet
    
    @staticmethod
    def from_dict(model_name, vec_l, vec_p, vec_d):
        return ANNVectorTriplet(
            model_name, 
            ANNVector.from_dict(model_name, "l", vec_l),
            ANNVector.from_dict(model_name, "p", vec_p),
            ANNVector.from_dict(model_name, "d", vec_d)
        )
    
    def to_dict(self):
        return self.vector_l, self.vector_p, self.vector_p
    
    def __repr__(self) -> str:
        return str({"model_name": self.model_name, "content": {"l": self.vector_l, "p": self.vector_p, "d": self.vector_d}})

class ANNVectorTripletArchGroup():

    def __init__(
        self,
        arch_name: str,
        vector_triplet_list: List[ANNVectorTriplet]
    ):
        self.arch_name = arch_name
        self.vector_triplet_list = vector_triplet_list

    @staticmethod
    def from_dataset(dataset: ClusterDataset, arch_name: str):

        if arch_name not in dataset.vec_l:
            return None

        raw_dict_l = dataset.get("l", arch_name, "arch")[arch_name]
        raw_dict_p = dataset.get("p", arch_name, "arch")[arch_name]
        raw_dict_d = dataset.get("d", arch_name, "arch")[arch_name]

        triplet_list = []
        for model_name in raw_dict_l.keys():
            curr_triplet = ANNVectorTriplet(
                model_name,
                ANNVector.from_dict(model_name, "l", raw_dict_l[model_name]),
                ANNVector.from_dict(model_name, "p", raw_dict_p[model_name]),
                ANNVector.from_dict(model_name, "d", raw_dict_d[model_name]),
            )
            triplet_list.append(curr_triplet)
        return ANNVectorTripletArchGroup(arch_name, triplet_list)
    
    def add(self, triplet: ANNVectorTriplet):
        self.vector_triplet_list.append(triplet)

    def get(self, i):
        return self.vector_triplet_list[i]
    
    def size(self):
        return len(self.vector_triplet_list)
    
    def to_dict(self):
        vec_l, vec_p, vec_d = {self.arch_name: dict()}, {self.arch_name: dict()}, {self.arch_name: dict()}
        for vector_triplet in self.vector_triplet_list:
            model_name = vector_triplet.model_name
            vec_l[self.arch_name][model_name] = vector_triplet.vector_l.content
            vec_p[self.arch_name][model_name] = vector_triplet.vector_p.content
            vec_d[self.arch_name][model_name] = vector_triplet.vector_d.content
        return vec_l, vec_p, vec_d
    
    @staticmethod
    def from_dict(vec_l, vec_p, vec_d):
        arch_name = list(vec_l.keys())[0]
        triplet_list = []
        for model_name in vec_l[arch_name].keys():
            curr_triplet = ANNVectorTriplet(
                model_name,
                ANNVector.from_dict(model_name, "l", vec_l[arch_name][model_name]),
                ANNVector.from_dict(model_name, "p", vec_p[arch_name][model_name]),
                ANNVector.from_dict(model_name, "d", vec_d[arch_name][model_name]),
            )
            triplet_list.append(curr_triplet)
        return ANNVectorTripletArchGroup(arch_name, triplet_list)
    
    def to_array(self):
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
        
        vec_l, vec_p, vec_d = {self.arch_name: dict()}, {self.arch_name: dict()}, {self.arch_name: dict()}
        for vector_triplet in self.vector_triplet_list:
            curr_vecarr_l, curr_vecarr_p, curr_vecarr_d = [0 for i in range(len(keys_l))], [0 for i in range(len(keys_p))], [0 for i in range(len(keys_d))]
            def fill_pure_int_vec(keys_arr, vec_arr, vec_triplet_content):
                for i in range(len(keys_arr)):
                    curr_key = keys_arr[i]
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
        self.vector_triplet_list = [triplet for triplet in self.vector_triplet_list if triplet.model_name != item_name]
    
    def __repr__(self) -> str:
        return str(self.vector_triplet_list)

if __name__ == "__main__":
    ds = ClusterDataset()
    arch_group = ANNVectorTripletArchGroup.from_dataset(ds, "WavLMForCTC")
    d_arch_group = arch_group.to_dict()
    print(arch_group)
    arch_group_recnstr = ANNVectorTripletArchGroup.from_dict(*d_arch_group)
    print(arch_group_recnstr)