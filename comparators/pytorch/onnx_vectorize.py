from list_gen import OrderedListGenerator
from auto_vectorizer import auto_vectorize
import onnx, json, os, fnmatch, pickle

def find_files(directory, pattern):
    l_fn, l_bn = [], []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                l_fn.append(filename)
                l_bn.append(basename[:-5])
    return l_fn, l_bn

def vectorize(path):
    print('vectorizing', path)
    m1 = onnx.load(path)
    gen1 = OrderedListGenerator(model=m1, mode='onnx', use_hash=True)
    l, c = gen1.get_connection()
    _, p, l, d = auto_vectorize(l, c, mode='onnx')
    return d, l, p

def auto_vectorize_from_model_pickle():

    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_p.json') as f:
        vec_p = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_d.json') as f:
        vec_d = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_l.json') as f:
        vec_l = json.load(f)
    
    def get_key_set(d):
        k_set = set()
        for k, v in d.items():
            for kk, vv in v.items():
                for kkk, vvv in vv.items():
                    k_set.add(kkk)
        return sorted(list(k_set))

    def create_default_list(ks):
        return [0 for key in ks]
    
    def add_padding(vec, ks):
        k2i_map = dict()
        for i, k in enumerate(ks):
            k2i_map[k] = i
        padded = dict()
        for arch, model_dict in vec.items():
            new_model_dict = dict()
            for model_name, model_vec in model_dict.items():
                new_model_vec = create_default_list(ks)
                for k, v in model_vec.items():
                    new_model_vec[k2i_map[k]] += v
                new_model_dict[model_name] = new_model_vec
            padded[arch] = new_model_dict
        return padded
                    
    
    k_p = get_key_set(vec_p)
    k_d = get_key_set(vec_d)
    k_l = get_key_set(vec_l)

    
    p_vec_l = add_padding(vec_l, k_l)
    p_vec_d = add_padding(vec_d, k_d)
    p_vec_p = add_padding(vec_p, k_p)
    

    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_l.pkl', 'wb') as f:
        pickle.dump(p_vec_l, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_d.pkl', 'wb') as f:
        pickle.dump(p_vec_d, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_p.pkl', 'wb') as f:
        pickle.dump(p_vec_p, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/k_l.pkl', 'wb') as f:
        pickle.dump(k_l, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/k_d.pkl', 'wb') as f:
        pickle.dump(k_d, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/k_p.pkl', 'wb') as f:
        pickle.dump(k_p, f)

def read_vec():
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_d.json', 'r') as f:
        df = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_l.json', 'r') as f:
        lf = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_p.json', 'r') as f:
        pf = json.load(f)
    return df, lf, pf

def write_vec(d, l, p, model_hub_type, model_name):
    df, lf, pf = read_vec()
    if model_hub_type not in df:
        df[model_hub_type] = dict()
        lf[model_hub_type] = dict()
        pf[model_hub_type] = dict()
    df[model_hub_type][model_name] = d
    lf[model_hub_type][model_name] = l
    pf[model_hub_type][model_name] = p
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_d.json', 'w') as f:
        json.dump(df, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_l.json', 'w') as f:
        json.dump(lf, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/onnx_vec/vec_p.json', 'w') as f:
        json.dump(pf, f)


    

lfn, lbn = find_files('/scratch/gilbreth/cheung59/PTMTorrent/PTMTorrent/ptm_torrent/onnxmodelzoo','*.onnx')

for i in range(len(lfn)):
    d, l, p = vectorize(lfn[i])
    write_vec(d, l, p, 'OnnxModelZoo', lbn[i])
    #print(lbn[i], len(p.keys()), sum([len(k) for k in p.keys()]))

print('2pickle in progress')
auto_vectorize_from_model_pickle()