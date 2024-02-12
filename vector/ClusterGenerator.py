
from .ClusterDataset import ClusterDataset
import numpy as np
import json
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import re
from .ANNVector import ANNVectorTripletArchGroup

class ClusterGenerator():

    def __init__(self, cluster_data_handler: ClusterDataset):
        self.cluster_data_handler = cluster_data_handler

    @staticmethod
    def concatenate_vec(dim_vec, layer_vec, param_vec, weights=[0, 1, 0.1], mode='internal'):
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

                # if model_arch not in model_vec:
                #     model_vec[model_arch] = {}
                    
                for model_name in models_dim_vec:
                    dim_arr = np.array(models_dim_vec[model_name])
                    layer_arr = np.array(models_layer_vec[model_name])
                    param_arr = np.array(models_param_vec[model_name])
                    #print(dim_arr, layer_arr, param_arr)
                    model_vec[model_family][model_name] = np.concatenate(
                        (weight_d * dim_arr, weight_l * layer_arr, weight_p * param_arr))
                    
        return model_vec

    def cluster_hf(self, weights=[0, 1, 0.1], eps=0.03):
        logger.info(f"Clustering model with eps={eps}!")
        return self.model_clustering(self.cluster_data_handler.load_pkl(), eps=eps, weights=weights)

    def model_clustering(self, model_vec, eps=0.8):
        results = {}
        outliers = {}
        # Initialize PCA
        pca = PCA(n_components=2)

        EPS = eps

        for model_family in tqdm(model_vec):

            results[model_family] = {}
            outliers[model_family] = []
            # Create a larger plot
            plt.figure(figsize=(10, 10))
            # if model_arch != "AlbertForMaskedLM":
            #     continue
            # logger.info(f"Clustering {len(model_vec[model_arch])} models in {model_arch}...")
            data = list(model_vec[model_family].values())

            model_names = list(model_vec[model_family].keys())
            if len(data) < 2:
                logger.warning(f"Not enough data to cluster model family {model_family}")
                continue

            # Standardize data to have a mean of ~0 and a variance of 1
            X_std = StandardScaler().fit_transform(data)


            # Check for zero variance features and remove them
            non_zero_var_indices = np.var(X_std, axis=0) != 0
            X_filtered = X_std[:, non_zero_var_indices]

            try:
                # Perform PCA
                data_pca = pca.fit_transform(X_filtered)
                # logger.debug(data_pca)
                    

                # Compute the cosine distances
                # cosine_distances = pairwise_distances(data_pca, metric='cosine')
                euclidean_distances = pairwise_distances(data_pca, metric='euclidean')
                # logger.debug(cosine_distances)
                # Perform DBSCAN on the data
                db = DBSCAN(eps=EPS, min_samples=2, metric='precomputed').fit(euclidean_distances)


                labels = db.labels_

                # Identify core samples
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True

                # Number of clusters in labels, ignoring noise if present.
                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                # Plot result
                unique_labels = set(labels)
                # logger.debug(f"Unique labels: {unique_labels}")
                colors = [plt.cm.Spectral(each)
                        for each in np.linspace(0, 1, len(unique_labels))]

                cluster_names = {}
                # Annotate points for both core and non-core
                for k, col in zip(unique_labels, colors):
                    class_member_mask = (labels == k)
                    xy = data_pca[class_member_mask]

                    if k != -1: 
                        cluster_names[k] = model_names[class_member_mask.tolist().index(True)]
                        centroid = np.mean(xy, axis=0)
                        plt.text(centroid[0]-1, centroid[1]+0.5, cluster_names[k], fontsize=15)

                        
                        # Initialize the list for the cluster if not already done
                        if k not in results[model_family]:
                            results[model_family][str(k)] = []
                        
                        # Save the model names for the cluster
                        for idx in np.where(class_member_mask)[0]:
                            results[model_family][str(k)].append(model_names[idx])


                    if k == -1:
                        col = [0, 0, 0, 1]
                        for idx, (x, y) in zip(np.where(class_member_mask)[0], xy):
                            plt.text(x-1, y+0.1, model_names[idx], fontsize=15, color='black')
                            outliers[model_family].append(model_names[idx])
                    
            except:
                # logger.warning(f"No variance in model family {model_family}")
                if not results[model_family]:
                    results[model_family]['0'] = []
                results[model_family]['0'].extend(model_names)
            
        return results, outliers
    
if __name__ == "__main__":
    ds = ClusterDataset()
    vtg0 = ANNVectorTripletArchGroup.from_dataset(ds, "Wav2Vec2Model")
    # WavLMForCTC
    extra = vtg0.get(0)
    vtg1 = ANNVectorTripletArchGroup.from_dataset(ds, "WavLMForCTC")
    print(vtg1.size())
    print(vtg1)
    vtg1.add(extra)
    print(vtg1)
    print(vtg1.size())
