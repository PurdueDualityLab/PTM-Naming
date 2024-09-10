"""
This module is used to cluster the model vectors using DBSCAN.
"""

from typing import List, Optional
import re
import numpy as np
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from vector.cluster_dataset import ClusterDataset
from vector.ann_vector import ANNVectorTripletArchGroup

class ClusterGenerator():
    """
    This class is used to cluster the model vectors using DBSCAN.

    Attributes:
        cluster_data_handler: The handler for the cluster dataset
    """
    def __init__(self, cluster_data_handler: ClusterDataset):
        self.cluster_data_handler = cluster_data_handler

    @staticmethod
    def concatenate_vec(
        dim_vec: dict,
        layer_vec: dict,
        param_vec: dict,
        weights: Optional[List] = None,
        mode = 'internal',
        remove_zero_weight: bool = True,
        verbose = False
    ) -> dict:
        """
        This function concatenates the vectors with weights.

        Args:
            dim_vec: The vector representation of the dimensions
            layer_vec: The vector representation of the layers
            param_vec: The vector representation of the parameters
            weights: The weights for the concatenation of the vectors
            mode: The mode of the vector representation. Choose from internal or external
            remove_zero_weight: Whether to remove the zero weight vectors
            verbose: Whether to print the progress

        Returns:
            The concatenated vectors
        """

        if weights is None:
            weights = [0, 1, 0.1]

        if verbose:
            logger.info(f"Concatenating vectors with weights [d, l, p]: {weights}...")
        # concatenate three vectors
        model_vec = {}

        weight_d = weights[0]
        weight_l = weights[1]
        weight_p = weights[2]

        if mode == 'internal':
            for model_arch in dim_vec:
                model_family = re.split('For|Model|LMHead', model_arch)[0]
                if model_family not in model_vec:
                    model_vec[model_family] = {}
                models_dim_vec = dim_vec[model_arch]
                models_layer_vec = layer_vec[model_arch]
                models_param_vec = param_vec[model_arch]

                for model_name in models_dim_vec:
                    dim_arr = np.array(models_dim_vec[model_name]) \
                        if weight_d != 0 or not remove_zero_weight else np.array([])
                    layer_arr = np.array(models_layer_vec[model_name]) \
                        if weight_l != 0 or not remove_zero_weight else np.array([])
                    param_arr = np.array(models_param_vec[model_name]) \
                        if weight_p != 0 or not remove_zero_weight else np.array([])
                    model_vec[model_family][model_name] = np.concatenate(
                        (weight_d * dim_arr, weight_l * layer_arr, weight_p * param_arr))

        return model_vec

    def model_clustering(
        self,
        model_vec: dict,
        eps: float = 0.8
    ):
        """
        This function clusters the model vectors using DBSCAN.

        Args:
            model_vec: The model vectors
            eps: The maximum distance between two samples for one to be 
                considered as in the neighborhood of the other
            verbose: Whether to print the progress

        Returns:
            The clustering results
        """
        results = {}
        outliers = {}
        # Initialize PCA
        pca = PCA(n_components=2)

        for model_family in model_vec:

            results[model_family] = {}
            outliers[model_family] = []

            data = list(model_vec[model_family].values())

            model_names = list(model_vec[model_family].keys())
            if len(data) < 2:
                logger.warning(f"Not enough data to cluster model family {model_family}")
                continue

            # Standardize data to have a mean of ~0 and a variance of 1
            x_std = StandardScaler().fit_transform(data) # type: ignore

            # Check for zero variance features and remove them
            non_zero_var_indices = np.var(x_std, axis=0) != 0
            x_filtered = x_std[:, non_zero_var_indices]

            try:
                # Perform PCA
                data_pca = pca.fit_transform(x_filtered)

                # Compute the cosine distances
                # cosine_distances = pairwise_distances(data_pca, metric='cosine')
                euclidean_distances = pairwise_distances(data_pca, metric='euclidean')
                # logger.debug(cosine_distances)
                # Perform DBSCAN on the data
                db = DBSCAN(eps=eps, min_samples=2, metric='precomputed').fit(euclidean_distances)

                labels = db.labels_

                # Identify core samples
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True

                # Plot result
                unique_labels = set(labels)

                cluster_names = {}
                # Annotate points for both core and non-core
                for k in unique_labels: # type: ignore
                    class_member_mask = labels == k

                    if k != -1:
                        cluster_names[k] = model_names[class_member_mask.tolist().index(True)]

                        # Initialize the list for the cluster if not already done
                        if k not in results[model_family]:
                            results[model_family][str(k)] = []

                        # Save the model names for the cluster
                        for idx in np.where(class_member_mask)[0]:
                            results[model_family][str(k)].append(model_names[idx])

                    if k == -1:
                        for idx in np.where(class_member_mask)[0]:
                            outliers[model_family].append(model_names[idx])

            except Exception: # pylint: disable=broad-except
                # logger.warning(f"No variance in model family {model_family}")
                if not results[model_family]:
                    results[model_family]['0'] = []
                results[model_family]['0'].extend(model_names)

        return results, outliers

if __name__ == "__main__":
    ds = ClusterDataset()
    vtg0 = ANNVectorTripletArchGroup.from_dataset(ds, "Wav2Vec2Model")
    assert vtg0 is not None
    # WavLMForCTC
    extra = vtg0.get(0)
    vtg1 = ANNVectorTripletArchGroup.from_dataset(ds, "WavLMForCTC")
    assert vtg1 is not None
    print(vtg1.size())
    print(vtg1)
    vtg1.add(extra)
    print(vtg1)
    print(vtg1.size())
