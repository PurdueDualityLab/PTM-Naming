
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_l.json') as f:
    data_l = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_p.json') as f:
    data_p = json.load(f)

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_d.json') as f:
    data_d = json.load(f)

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
    ret = dict()
    for n, v in data.items():
        if n == benchmark: continue
        curr_vec = v
        s = get_norm_ed(benchmark_vec, curr_vec)
        ret[n] = s
        #print(n + ':', s)
    return ret

def name_mod(n):
    new_n = ''
    for ch in n:
        nch = ch
        if ch == '/':
            nch = '>'
        new_n += nch
    return new_n

arch_subtypes = dict()
for arch_type in data_l.keys():
    info_l, info_p, info_d = data_l[arch_type], data_p[arch_type], data_d[arch_type]
    arch_type_model_count = len(info_l.keys())
    grouped_model = set()
    groups = []
    while len(grouped_model) != arch_type_model_count:
        for model_name in info_l.keys(): # look for a model thats not in a group
            if model_name not in grouped_model:
                benchmark_model = model_name
                break
        grouped_model.add(benchmark_model)
        curr_group = {benchmark_model}
        dict_l = calc_norm_euc_dist(info_l, benchmark_model)
        dict_p = calc_norm_euc_dist(info_p, benchmark_model)
        dict_d = calc_norm_euc_dist(info_d, benchmark_model)
        for model_name in dict_l.keys():

            f1 = open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_data/' + name_mod(model_name) + '.json')
            f2 = open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_data/' + name_mod(model_name) + '.json')
            d1 = json.load(f1)
            d2 = json.load(f2)
            f1.close()
            f2.close()

            if dict_l[model_name] + dict_p[model_name] + dict_d[model_name] >= 3 and d1 == d2:
                grouped_model.add(model_name)
                curr_group.add(model_name)
        groups.append(list(curr_group))
    arch_subtypes[arch_type] = groups

with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/absolute_groups.json', 'w') as f:
    json.dump(arch_subtypes, f)

