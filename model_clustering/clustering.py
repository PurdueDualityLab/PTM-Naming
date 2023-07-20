import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def load_json_files(filenames):
    json_contents = []
    for filename in filenames:
        with open(filename, 'r') as f:
            json_contents.append(json.load(f))
    return json_contents


def extract_descriptions(json_files):
    descriptions = []
    for json_file in json_files:
        for key, value in json_file.items():
            for sub_key, sub_value in value.items():
                descriptions.append(str(sub_value))
    return descriptions


def cluster_and_visualize(descriptions):
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(descriptions).toarray()
    reduced_data = PCA(n_components=2).fit_transform(X)
    dbscan = DBSCAN(eps=0.3, min_samples=20).fit(reduced_data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=dbscan.labels_)
    plt.savefig("clustering.png")


if __name__ == "__main__":
    json_files = load_json_files(["../comparators/pytorch/ptm_vectors/vec_d.json", "../comparators/pytorch/ptm_vectors/vec_l.json", "../comparators/pytorch/ptm_vectors/vec_p.json"])
    descriptions = extract_descriptions(json_files)
    cluster_and_visualize(descriptions)
