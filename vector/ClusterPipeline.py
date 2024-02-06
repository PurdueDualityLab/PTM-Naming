
from ANNVector import *
from ClusterGenerator import *
from ClusterDataset import *

class ClusterPipeline():

    def __init__(self):
        self.cluster_data = ClusterDataset()

    def cluster_with_extra_model(
        self,
        arch_name: str,
        additional_model_vector: ANNVectorTriplet,
        eps: int = 0.3
    ):
        model_vector_group = ANNVectorTripletArchGroup.from_dataset(self.cluster_data, arch_name)
        model_vector_group.add(additional_model_vector)
        #print(model_vector_group)
        vec_l, vec_p, vec_d = model_vector_group.to_array()
        #print(vec_l)
        model_vec = ClusterGenerator.concatenate_vec(vec_d, vec_l, vec_p)
        #print(model_vec)
        results, outliers = ClusterGenerator(self.cluster_data).model_clustering(model_vec, eps=eps)
        return results, outliers
    
if __name__ == "__main__":
    ds = ClusterDataset()
    vtg0 = ANNVectorTripletArchGroup.from_dataset(ds, "Wav2Vec2Model")
    extra = vtg0.get(0)
    res, out = ClusterPipeline().cluster_with_extra_model("WavLMForCTC", extra)
    print(res, out)

