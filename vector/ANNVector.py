
from typing import List
from ClusterDataset import ClusterDataset
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
    
    def __repr__(self) -> str:
        return str(self.vector_triplet_list)

if __name__ == "__main__":
    ds = ClusterDataset()
    arch_group = ANNVectorTripletArchGroup.from_dataset(ds, "WavLMForCTC")
    print(arch_group)