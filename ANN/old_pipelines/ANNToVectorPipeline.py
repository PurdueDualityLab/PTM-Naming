
from transformers import AutoModel, AutoTokenizer
import json
from ANN.ann_generator import AbstractNNGenerator
from ANN.utils import overwrite_torchview_func
import os
from transformers import AutoModel, AutoTokenizer
from ANN.ann_layer import AbstractNNLayer
from typing import List, Tuple
from ANN.pipelines.ANNToJSONConverter import read_annlayer_list_from_json
import pickle

# Set the cache directory
os.environ["HF_HOME"] = "/scratch/gilbreth/cheung59/cache_huggingface"
d = "/scratch/gilbreth/cheung59/cache_huggingface"

def read_model_from_json(dir, target_arch_family):
    with open(dir) as f:
        data = json.load(f)
    l = data[target_arch_family]
    return l
def read_json(dir):
    with open(dir) as f:
        data = json.load(f)
    return data

def auto_vectorize_nlp(model_name_list, output_dir_l, output_dir_p, output_dir_pl, output_dir_d, output_dir_dn):
    overwrite_torchview_func()
    #d_l, d_p, d_pl, d_d, d_dn = read_json(output_dir_l), read_json(output_dir_p), read_json(output_dir_pl)
    d_l, d_p, d_pl, d_d, d_dn = dict(), dict(), dict(), dict(), dict()
    for n in model_name_list:
        if n in d_l.keys():
            print(n, 'already exists')
            continue
        print('Working on vectorizing', n)
        try:
            t = AutoTokenizer.from_pretrained(n, cache_dir = d)
            m = AutoModel.from_pretrained(n, cache_dir = d)
        except Exception as e:
            print('Failed to vectorize:', n)
            print(e)
            continue
        inp = t("Test Input", return_tensors="pt")
        gen = AbstractNNGenerator(m, inp, use_hash=True)
        fvl, fvp, fvpl, fvd, fvdn = gen.vectorize()
        d_l[n] = fvl
        d_p[n] = fvp
        d_pl[n] = fvpl
        d_d[n] = fvd
        d_dn[n] = fvdn
        with open(output_dir_l, 'w') as f:
            json.dump(d_l, f)
        with open(output_dir_p, 'w') as f:
            json.dump(d_p, f)
        with open(output_dir_pl, 'w') as f:
            json.dump(d_pl, f)
        with open(output_dir_d, 'w') as f:
            json.dump(d_d, f)
        with open(output_dir_dn, 'w') as f:
            json.dump(d_dn, f)
        print('Finished on vectorizing', n)


'''
l = read_model_from_json('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/temp_models.json', "BertForMaskedLM")
auto_vectorize_nlp(
    l, 
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_l.json', 
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_p.json', 
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_pl.json',
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_d.json',
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_dn.json'
    )
    '''
def auto_vectorize(l_l, c_i, mode='pytorch'):

    def get_freq_vec_l(layer_list, connection_info):
        freq_vec = dict()
        id_to_node_map = dict()
        for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
            id_to_node_map[layer_connection_info[0]] = layer_node
            
        def make_node_string(n: AbstractNNLayer):
            if n.is_input_node:
                if mode == 'pytorch': return '[INPUT] {}'.format(n.output_shape)
                if mode == 'onnx': return '[INPUT]'
            if n.is_output_node:
                if mode == 'pytorch': return '[OUTPUT] {}'.format(n.input_shape)
                if mode == 'onnx': return '[OUTPUT]'
            if mode == 'pytorch': return '{} {}->{}'.format(n.operation, n.input_shape, n.output_shape)
            if mode == 'onnx': 
                if len(n.input_shape) == 0:
                    return '{}'.format(n.operation)
                return '{} {}'.format(n.operation, n.input_shape)
            
        for layer_node, layer_connection_info in zip(layer_list, connection_info):
            curr_node_str = make_node_string(layer_node)
            for next_layer_id in layer_connection_info[1]:
                next_node_str = make_node_string(id_to_node_map[next_layer_id])
                combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                if combined_str not in freq_vec:
                    freq_vec[combined_str] = 0
                freq_vec[combined_str] += 1
        
        return freq_vec
    
    def check_long_str(s):
        s = str(s)
        if len(s) > 500:
            s = 'long str: ' + str(hash(s))
        return s

    def get_freq_vec_p(layer_list: List[AbstractNNLayer]):
        freq_vec = dict()
        for l in layer_list:
            p_str_list = []
            if l.parameters != None:
                for p in l.parameters:
                    p_str_list.append('<{}, {}>'.format(p.param_name, p.param_value))
            p_str_list = check_long_str(p_str_list)
            if l.is_input_node:
                l_str = '[INPUT]'
            elif l.is_output_node:
                l_str = '[OUTPUT]'
            else:
                l_str = '{} {}'.format(l.operation, p_str_list if len(p_str_list) else '')
            if l_str not in freq_vec:
                freq_vec[l_str] = 0
            freq_vec[l_str] += 1
        return freq_vec
    
    def get_freq_vec_pl(layer_list, connection_info):
        freq_vec = dict()
        id_to_node_map = dict()
        for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
            id_to_node_map[layer_connection_info[0]] = layer_node
            
        def make_node_string(n: AbstractNNLayer):
            if n.is_input_node:
                return '[INPUT]'
            if n.is_output_node:
                return '[OUTPUT]'
            return n.operation

        for layer_node, layer_connection_info in zip(layer_list, connection_info):
            curr_node_str = make_node_string(layer_node)
            for next_layer_id in layer_connection_info[1]:
                next_node_str = make_node_string(id_to_node_map[next_layer_id])
                combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                if combined_str not in freq_vec:
                    freq_vec[combined_str] = 0
                freq_vec[combined_str] += 1
        
        return freq_vec
    
    def get_freq_vec_d(layer_list):
        freq_vec_d = dict()
        for l in layer_list:
            d_list = []
            if l.input_shape != None:
                for s_in in l.input_shape: 
                    d_list.append(s_in)
            if l.output_shape != None:
                for s_out in l.output_shape: 
                    d_list.append(s_out)
            for t in d_list:
                if str(t) not in freq_vec_d:
                    freq_vec_d[str(t)] = 0
                freq_vec_d[str(t)] += 1
            
        return freq_vec_d
    
    fv_l = get_freq_vec_l(l_l, c_i)
    fv_p = get_freq_vec_p(l_l)
    fv_pl = get_freq_vec_pl(l_l, c_i)
    fv_d = get_freq_vec_d(l_l)

    return fv_l, fv_p, fv_pl, fv_d

def auto_vectorize_from_model_json(input_dir, models_dict, output_dir_l, output_dir_p, output_dir_d):
    d_l, d_p, d_d = dict(), dict(), dict()
    with open(output_dir_l) as f:
        d_l = json.load(f)
    with open(output_dir_p) as f:
        d_p = json.load(f)
    with open(output_dir_d) as f:
        d_d = json.load(f)
    for model_arch in models_dict.keys():
        for model_name in models_dict[model_arch]:
            if model_name == None:
                continue
            if model_arch in d_l.keys():
                if model_name in d_l[model_arch].keys():
                    print('Skipped file', input_dir + '/' + model_name + '.json:')
                    continue
            new_model_name = ''
            with open(output_dir_l) as f:
                d_l = json.load(f)
            with open(output_dir_p) as f:
                d_p = json.load(f)
            with open(output_dir_d) as f:
                d_d = json.load(f)
            if model_arch not in d_l.keys():
                d_l[model_arch], d_p[model_arch], d_d[model_arch] = dict(), dict(), dict()
            for ch in model_name:
                nch = ch
                if ch == '/':
                    nch = '>'
                new_model_name += nch 
            try:
                l_l, c_i = read_annlayer_list_from_json(input_dir + '/' + new_model_name + '.json')
            except Exception as e:
                print('Cannot read file', input_dir + '/' + model_name + '.json:', e)
                if d_l[model_arch] == dict():
                    del d_l[model_arch]
                    del d_p[model_arch]
                    del d_d[model_arch]
                continue
            print('Vectorizing', input_dir + '/' + model_name + '.json')
            fvs = auto_vectorize(l_l, c_i)
            d_l[model_arch][model_name] = fvs[2]
            d_p[model_arch][model_name] = fvs[1]
            d_d[model_arch][model_name] = fvs[3]
            with open(output_dir_l, 'w') as f:
                json.dump(d_l, f)
            with open(output_dir_p, 'w') as f:
                json.dump(d_p, f)
            with open(output_dir_d, 'w') as f:
                json.dump(d_d, f)

def auto_vectorize_from_model_pickle(
    
    ):
    '''
    with open(output_dir_l) as f:
        d_l = pickle.load(f)
    with open(output_dir_p) as f:
        d_p = pickle.load(f)
    with open(output_dir_d) as f:
        d_d = pickle.load(f)
    with open(output_dir_lk) as f:
        d_lk = pickle.load(f)
    with open(output_dir_pk) as f:
        d_pk = pickle.load(f)
    with open(output_dir_dk) as f:
        d_dk = pickle.load(f)
    '''
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_p.json') as f:
        vec_p = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_d.json') as f:
        vec_d = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_l.json') as f:
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
    

    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_l.pkl', 'wb') as f:
        pickle.dump(p_vec_l, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_d.pkl', 'wb') as f:
        pickle.dump(p_vec_d, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_p.pkl', 'wb') as f:
        pickle.dump(p_vec_p, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/k_l.pkl', 'wb') as f:
        pickle.dump(k_l, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/k_d.pkl', 'wb') as f:
        pickle.dump(k_d, f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/k_p.pkl', 'wb') as f:
        pickle.dump(k_p, f)

'''
with open('/depot/davisjam/data/chingwo/PTM-Naming/model_collection/filtered_models.json') as f:
        models_dict = json.load(f)

auto_vectorize_from_model_json(
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_data',
    models_dict,
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_l.json',
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_p.json',
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/vec_d.json'
    )
'''

auto_vectorize_from_model_pickle()