import json
import os
import pickle
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import great_circle
from loguru import logger
from matplotlib.lines import Line2D
from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

# load HF internal vectors
D_PATH = "./vectors/vec_d.pkl"
L_PATH = "./vectors/vec_l.pkl"
P_PATH = "./vectors/vec_p.pkl"

# load external vectors 
ONNX_D_PATH = "../vectors/vec_d.pkl"
ONNX_L_PATH = "../vectors/vec_l.pkl"
ONNX_P_PATH = "../vectors/vec_p.pkl"

def load_vec(path):
    # load pickle file from path
    with open(path, 'rb') as f:
        return pickle.load(f)


def concatenate_vec(dim_vec, layer_vec, param_vec):
    # concatenate three vectors
    model_vec = {}
    for model_arch in dim_vec:
        models_dim_vec = dim_vec[model_arch]
        models_layer_vec = layer_vec[model_arch]
        models_param_vec = param_vec[model_arch]

        if model_arch not in model_vec:
            model_vec[model_arch] = {}

        for model_name in models_dim_vec:
            dim_arr = np.array(models_dim_vec[model_name])
            layer_arr = np.array(models_layer_vec[model_name])
            param_arr = np.array(models_param_vec[model_name])
            model_vec[model_arch][model_name] = np.concatenate(
                (dim_arr, layer_arr, param_arr))
    return model_vec


def save_vec(model_vec):
    with open('model_vec.pkl', 'wb') as f:
        pickle.dump(model_vec, f)


def model_clustering(model_vec, eps=0.8):
    # Initialize PCA
    pca = PCA(n_components=2)

    EPS = eps
    if not os.path.exists(f"Results_{EPS}"):
        os.makedirs(f"Results_{EPS}")

    for model_arch in tqdm(model_vec):
        # Create a larger plot
        plt.figure(figsize=(10, 10))
        # if model_arch != "AlbertForMaskedLM":
        #     continue
        # logger.info(f"Clustering {len(model_vec[model_arch])} models in {model_arch}...")
        data = list(model_vec[model_arch].values())
        model_names = list(model_vec[model_arch].keys())
        if len(data) < 2:
            logger.warning(f"{model_arch}: Not enough data to cluster")
            continue

        # Standardize data to have a mean of ~0 and a variance of 1
        X_std = StandardScaler().fit_transform(data)
        # Perform PCA
        data_pca = pca.fit_transform(X_std)

        # Perform DBSCAN on the data
        db = DBSCAN(eps=EPS, min_samples=2).fit(data_pca)

        labels = db.labels_

        # Identify core samples
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        # Plot result
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        cluster_names = {}
        # Annotate points for both core and non-core
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = data_pca[class_member_mask]

            # Assign first model name to each cluster
            if k != -1:
                cluster_names[k] = model_names[class_member_mask.tolist().index(
                    True)]
                centroid = np.mean(xy, axis=0)
                plt.text(centroid[0]-1, centroid[1]+0.5,
                         cluster_names[k], fontsize=15)

            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                for idx, (x, y) in zip(np.where(class_member_mask)[0], xy):
                    plt.text(
                        x-1, y+0.1, model_names[idx], fontsize=15, color='black')

            class_member_mask = (labels == k)
            xy = data_pca[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=20)

            xy = data_pca[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)

        plt.title(
            f'{model_arch}: {n_clusters_} clusters, {n_noise_} outliers', fontsize=30)
        plt.axis('off')

        # ################################################################################
        # # Plotting and legend creation
        # legend_elements = []
        # for i, col in enumerate(colors):
        #     if i in cluster_names:  # Add only if the cluster is in the cluster_names dictionary
        #         legend_elements.append(Line2D([0], [0], marker='o', color='w',
        #                                     label=cluster_names[i],
        #                                     markerfacecolor=col, markersize=10))
        # # For outlier, create a Line2D with black color
        # legend_elements.append(Line2D([0], [0], marker='o', color='w',
        #                             label='Outlier', markerfacecolor='black', markersize=10))

        # # Legend
        # legend = plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.2, 0.8), fontsize=20)

        # # Change color of text for outliers
        # legend.get_texts()[-1].set_color("black")
        # ################################################################################

        plt.savefig(
            f"./Results_{EPS}/{model_arch}_{n_clusters_}clusters_{n_noise_}outliers.png", pad_inches=1)
        plt.close()
        # # Add legend
        # handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.Spectral(i/len(unique_labels)), markersize=8) for i in range(len(unique_labels))]
        # legend = plt.legend(handles, [cluster_names.get(i, 'Outlier') for i in range(len(unique_labels))], fontsize=15, bbox_to_anchor=(1.5, 1))
        # legend.get_texts()[0].set_color('black')
        # # logger.info(f"Saving to {model_arch}_{n_clusters_}clusters_{n_noise_}outliers.png")
        # plt.savefig(f"./Results_{EPS}/{model_arch}_{n_clusters_}clusters_{n_noise_}outliers.png", bbox_inches='tight')
        # plt.close()
        # exit()
    return


def model_clustering_all(model_vec, eps=0.8):
    # Initialize PCA
    pca = PCA(n_components=2)

    EPS = eps
    if not os.path.exists(f"Results_{EPS}"):
        os.makedirs(f"Results_{EPS}")

    # Create a larger plot
    plt.figure(figsize=(10, 10))
    # if model_arch != "AlbertForMaskedLM":
    #     continue
    # logger.info(f"Clustering {len(model_vec[model_arch])} models in {model_arch}...")
    data = []
    model_names = []
    for model_arch in tqdm(model_vec):
        data.extend(list(model_vec[model_arch].values()))
        model_names.extend(list(model_vec[model_arch].keys()))
    # logger.debug(model_names)
    # exit()
    # Standardize data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(data)
    # Perform PCA
    data_pca = pca.fit_transform(X_std)

    # Perform DBSCAN on the data
    db = DBSCAN(eps=EPS, min_samples=2).fit(data_pca)

    labels = db.labels_

    # Identify core samples
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Plot result
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    cluster_names = {}
    # Annotate points for both core and non-core
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = data_pca[class_member_mask]

        # # Assign first model name to each cluster
        # if k != -1:
        #     cluster_names[k] = model_names[class_member_mask.tolist().index(True)]
        #     centroid = np.mean(xy, axis=0)
        #     plt.text(centroid[0]-1, centroid[1]+0.5, cluster_names[k], fontsize=15)

        # if k == -1:
        #     # Black used for noise.
        #     col = [0, 0, 0, 1]
        #     for idx, (x, y) in zip(np.where(class_member_mask)[0], xy):
        #         plt.text(x-1, y+0.1, model_names[idx], fontsize=15, color='black')

        class_member_mask = (labels == k)
        xy = data_pca[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=20)

        xy = data_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=10)

    plt.title(
        f'All Models: {n_clusters_} clusters, {n_noise_} outliers', fontsize=30)
    plt.axis('off')
    logger.success(f'All Models: {n_clusters_} clusters, {n_noise_} outliers')
    plt.savefig(
        f"./All_{n_clusters_}clusters_{n_noise_}outliers.png", pad_inches=1)
    plt.close()
    return


def model_clustering_all_3D(model_vec, eps=0.8):
    # Initialize PCA
    pca = PCA(n_components=3)

    EPS = eps
    if not os.path.exists(f"Results_{EPS}"):
        os.makedirs(f"Results_{EPS}")

    # Preparing data for PCA and DBSCAN
    data = []
    model_names = []
    for model_arch in tqdm(model_vec):
        data.extend(list(model_vec[model_arch].values()))
        model_names.extend(list(model_vec[model_arch].keys()))

    # Standardize data to have a mean of ~0 and a variance of 1
    X_std = StandardScaler().fit_transform(data)

    # Perform PCA
    data_pca = pca.fit_transform(X_std)

    # Perform DBSCAN on the data
    db = DBSCAN(eps=EPS, min_samples=2).fit(data_pca)

    labels = db.labels_

    # Identify core samples
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Plot result
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    traces = []

    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = data_pca[class_member_mask & core_samples_mask]
        if k != -1: 
            # Non-outlier clusters
            model_name = model_names[class_member_mask.tolist().index(True)]
            trace_name = model_name
        else:
            # Outliers
            trace_name = 'Outlier'
            print("Outliers: ", model_names[class_member_mask.tolist().index(True)])

        traces.append(
            go.Scatter3d(
                x=xy[:, 0],
                y=xy[:, 1],
                z=xy[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    line=dict(
                        color='rgba(217, 217, 217, 0.14)',
                        width=0.5),
                    opacity=0.8),
                name=trace_name)
            )

    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

    return


def main(save_vec=False):
    if os.path.exists("model_vec.pkl"):
        with open('model_vec.pkl', 'rb') as f:
            model_vec = pickle.load(f)
    else:
        dim_vec = load_vec(D_PATH)
        layer_vec = load_vec(L_PATH)
        param_vec = load_vec(P_PATH)
        # logger.debug(dim_vec.keys())
        # logger.debug(layer_vec.keys())
        # logger.debug(param_vec.keys())

        model_vec = concatenate_vec(dim_vec, layer_vec, param_vec)
        # logger.debug(model_vec)
        if save_vec:
            save_vec(model_vec)
    # for eps in [0.2, 0.5, 0.7, 0.9]:
    eps = 0.8
    logger.info(f"Clustering model with eps={eps}!")
    # model_clustering(model_vec, eps=eps)
    model_clustering_all_3D(model_vec, eps=eps)


if __name__ == "__main__":
    json_files = load_json_files(["../comparators/pytorch/ptm_vectors/vec_d.json", "../comparators/pytorch/ptm_vectors/vec_l.json", "../comparators/pytorch/ptm_vectors/vec_p.json"])
    descriptions = extract_descriptions(json_files)
    cluster_and_visualize(descriptions)
