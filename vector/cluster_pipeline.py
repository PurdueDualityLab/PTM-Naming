"""
This module provides a pipeline for clustering models based on their vector representations.
"""
from typing import Tuple, Union
from transformers import PretrainedConfig
from loguru import logger
from vector.ann_vector import ANNVectorTriplet, ANNVectorTripletArchGroup, AbstractNN
from vector.cluster_generator import ClusterGenerator
from vector.cluster_dataset import ClusterDataset


class ClusterPipeline():
    """
    This class provides a pipeline for clustering models based on their vector representations.

    Attributes:
        cluster_data (ClusterDataset): The cluster dataset.
    """
    def __init__(self):
        self.cluster_data = ClusterDataset()

    def cluster_arch(
        self,
        arch_name: str,
        eps: float = 0.3
    ) -> Tuple[dict, dict]:
        """
        This function clusters the models of the input architecture.

        Args:
            arch_name (str): The name of the architecture.
            eps (float): The maximum distance between two samples for one to be 
                considered as in the neighborhood of the other.
        
        Returns:
            Tuple[dict, dict]: The clustered models and the outliers.
        """
        vec_l, vec_p, vec_d = \
            self.cluster_data.get("l", arch_name), \
            self.cluster_data.get("p", arch_name), \
            self.cluster_data.get("d", arch_name)
        model_vec = ClusterGenerator.concatenate_vec(vec_d, vec_l, vec_p)
        results, outliers = ClusterGenerator(self.cluster_data).model_clustering(model_vec, eps=eps)
        return results, outliers

    def cluster_with_extra_model(
        self,
        arch_name: str,
        additional_model_vector: ANNVectorTriplet,
        eps: float = 0.3
    ) -> Tuple[dict, dict]:
        """
        This function clusters the models of the input architecture with an additional model.

        Args:
            arch_name (str): The name of the architecture.
            additional_model_vector (ANNVectorTriplet): The vector 
                representation of the additional model.
            eps (float): The maximum distance between two samples for
                one to be considered as in the neighborhood of the other.
        
        Returns:
            Tuple[dict, dict]: The clustered models and the outliers.
        """
        model_vector_group = ANNVectorTripletArchGroup.from_dataset(self.cluster_data, arch_name)
        if model_vector_group is None:
            model_vector_group = ANNVectorTripletArchGroup(arch_name, [])
        model_vector_group.add(additional_model_vector)
        vec_l, vec_p, vec_d = model_vector_group.to_array()
        model_vec = ClusterGenerator.concatenate_vec(vec_d, vec_l, vec_p)
        results, outliers = ClusterGenerator(self.cluster_data).model_clustering(model_vec, eps=eps)
        return results, outliers

    def cluster_with_extra_model_from_huggingface(
        self,
        hf_repo_name: str,
        arch_name: str = "auto",
        model_name: str = "auto",
        eps: float = 0.3
    ) -> Tuple[dict, dict]:
        """
        This function clusters the models of the input architecture with an additional 
        model from Hugging Face.

        Args:
            hf_repo_name (str): The name of the Hugging Face repository.
            arch_name (str): The name of the architecture.
            model_name (str): The name of the model.
            eps (float): The maximum distance between two samples for one to be considered 
                as in the neighborhood of the other.
        
        Returns:
            Tuple[dict, dict]: The clustered models and the outliers.
        """
        if model_name == "auto":
            model_name = hf_repo_name
        if arch_name == "auto":
            arch_name = PretrainedConfig.from_pretrained(hf_repo_name).architectures[0]
            logger.info(f"Automatically identified architecture of hf model as {arch_name}.")
        ann = AbstractNN.from_huggingface(hf_repo_name)
        ann_vector_triplet = ANNVectorTriplet.from_ann(model_name, ann)
        return self.cluster_with_extra_model(arch_name, ann_vector_triplet, eps)

    def cluster_single_arch_from_dict(
        self,
        vec_dict_triplet: tuple,
        eps: float = 0.3,
        merge_outlier: bool = False
    ) -> Union[dict, Tuple[dict, dict]]:
        """
        This function clusters the models of the input architecture with an additional

        Args:
            vec_dict_triplet (tuple): The vector representation of the architecture.
            eps (float): The maximum distance between two samples for one to be considered 
                as in the neighborhood of the other.
            merge_outlier (bool): The merge outlier flag.
        
        Returns:
            Union[dict, Tuple[dict, dict]]: The clustered models and the outliers.
        """
        model_vec = self.get_model_vec_from_dict(vec_dict_triplet)
        results, outliers = ClusterGenerator(self.cluster_data).model_clustering(model_vec, eps=eps)

        if merge_outlier:
            arch_str = str(list(outliers.keys())[0])
            start_idx = len(list(results[arch_str].keys()))

            for model in outliers[arch_str]:
                results[arch_str][str(start_idx)] = [model]
                start_idx += 1

            return results

        return results, outliers
    
    def get_model_vec_from_dict(
        self,
        vec_dict_triplet: tuple
    ) -> dict:
        """
        This function returns the model vector from the input dictionary.

        Args:
            vec_dict_triplet (tuple): The vector representation of the architecture.
        
        Returns:
            dict: The model vector.
        """
        vec_l, vec_p, vec_d = ANNVectorTripletArchGroup.from_dict(
            vec_dict_triplet[0],
            vec_dict_triplet[1],
            vec_dict_triplet[2]
        ).to_array()
        return ClusterGenerator.concatenate_vec(vec_d, vec_l, vec_p)


if __name__ == "__main__":
    print(ClusterPipeline().cluster_with_extra_model_from_huggingface("microsoft/resnet-18"))
