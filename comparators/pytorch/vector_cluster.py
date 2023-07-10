
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_l.json') as f:
    data_l = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_p.json') as f:
    data_p = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_pl.json') as f:
    data_pl = json.load(f)

def get_cosine_similarity(fv1, fv2):
    keys = list(set(list(fv1.keys()) + list(fv2.keys())))

    vec1 = np.array([fv1.get(key, 0) for key in keys])
    vec2 = np.array([fv2.get(key, 0) for key in keys])

    cos_sim = cosine_similarity([vec1], [vec2])

    return cos_sim[0][0]

def get_norm_ed(fv1, fv2):
    keys = list(set(list(fv1.keys()) + list(fv2.keys())))

    vec1 = np.array([fv1.get(key, 0) for key in keys])
    vec2 = np.array([fv2.get(key, 0) for key in keys])

    ed = np.linalg.norm(vec2 - vec1)
    return 1 / (1 + ed)

def calc_sim(data, benchmark):
    benchmark_vec = data[benchmark]
    print('Benchmark:', benchmark)
    for n, v in data.items():
        if n == benchmark: continue
        curr_vec = v
        s = get_cosine_similarity(benchmark_vec, curr_vec)
        print(curr_vec)
        print(n + ':', s)

def calc_norm_euc_dist(data, benchmark):
    benchmark_vec = data[benchmark]
    print('Benchmark:', benchmark)
    for n, v in data.items():
        if n == benchmark: continue
        curr_vec = v
        s = get_norm_ed(benchmark_vec, curr_vec)
        print(n + ':', s)

calc_norm_euc_dist(data_pl, "bert-base-cased")