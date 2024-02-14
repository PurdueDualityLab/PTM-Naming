from ANN.ann_generator import AbstractNNGenerator
from ANN.pipelines.ANNToVectorPipeline import auto_vectorize
import onnx, json, os, fnmatch, pickle

def find_files(directory, pattern):
    """
    Find all the files in a directory given a specific pattern

    Parameters:
    directory (str): The target directory
    pattern (str): The pattern to look for

    Returns:
    l_fn, l_bn (tuple): A tuple of a list of file names and a list of base names (without extension)
    """
    l_fn, l_bn = [], []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.abspath(os.path.join(root, basename))
                l_fn.append(filename)
                l_bn.append(basename[:-5])
    return l_fn, l_bn

def vectorize(path):
    """
    Vectorize an ONNX model

    Parameters:
    path (str): The path of the ONNX model

    Returns:
    d, l, p (tuple): A tuple consists of dimension, parameter, and layer vectors
    of a single model
    """
    print('vectorizing', path)
    m1 = onnx.load(path)
    gen1 = AbstractNNGenerator(model=m1, framework='onnx', use_hash=True)
    l, c = gen1.get_connection()
    _, p, l, d = auto_vectorize(l, c, mode='onnx')
    return d, l, p

def pad_vector_to_pickle(directory):
    """
    Add padding to the vectors in a directory, and generates lists of keys and 
    present vector with arrays where the key each index is the content of the 
    same index in the key array

    Parameters:
    directory (str): The directory of the three vectors

    Returns:
    None
    """
    with open(f'{directory}/vec_p.json') as f:
        vec_p = json.load(f)
    with open(f'{directory}/vec_d.json') as f:
        vec_d = json.load(f)
    with open(f'{directory}/vec_l.json') as f:
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
    

    with open(f'{directory}/vec_l.pkl', 'wb') as f:
        pickle.dump(p_vec_l, f)
    with open(f'{directory}/vec_d.pkl', 'wb') as f:
        pickle.dump(p_vec_d, f)
    with open(f'{directory}/vec_p.pkl', 'wb') as f:
        pickle.dump(p_vec_p, f)
    with open(f'{directory}/k_l.pkl', 'wb') as f:
        pickle.dump(k_l, f)
    with open(f'{directory}/k_d.pkl', 'wb') as f:
        pickle.dump(k_d, f)
    with open(f'{directory}/k_p.pkl', 'wb') as f:
        pickle.dump(k_p, f)

def read_vec(directory):
    """
    Read the three vectors from a directory

    Parameters:
    directory (str): The directory of the three vectors

    Returns:
    df, lf, pf (tuple): A tuple consists of dimension, parameter, and layer vectors
    of all models
    """
    with open(f'{directory}/vec_d.json', 'r') as f:
        df = json.load(f)
    with open(f'{directory}/vec_l.json', 'r') as f:
        lf = json.load(f)
    with open(f'{directory}/vec_p.json', 'r') as f:
        pf = json.load(f)
    return df, lf, pf

def write_vec(d, l, p, model_hub_type, model_name, directory):
    """
    Write the three vectors to a directory. It appends the given three vectors
    of a single model to the three vectors in the directory that contains
    multiple models.

    Parameters:
    d (dict): The dimension vector
    l (dict): The layer vector
    p (dict): The parameter vector
    model_hub_type (str): The name of the model hub
    model_name (str): The name of the model
    directory (str): The directory containing the three vectors

    Returns:
    None
    """
    df, lf, pf = read_vec(directory)
    if model_hub_type not in df:
        df[model_hub_type] = dict()
        lf[model_hub_type] = dict()
        pf[model_hub_type] = dict()
    df[model_hub_type][model_name] = d
    lf[model_hub_type][model_name] = l
    pf[model_hub_type][model_name] = p
    with open(f'{directory}/vec_d.json', 'w') as f:
        json.dump(df, f)
    with open(f'{directory}/vec_l.json', 'w') as f:
        json.dump(lf, f)
    with open(f'{directory}/vec_p.json', 'w') as f:
        json.dump(pf, f)

