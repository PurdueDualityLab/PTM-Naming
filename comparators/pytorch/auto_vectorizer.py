
from transformers import AutoModel, AutoTokenizer
import json
from list_gen import OrderedListGenerator
from utils import patch
import os
from transformers import AutoModel, AutoTokenizer

# Set the cache directory
os.environ["HF_HOME"] = "/scratch/gilbreth/cheung59/cache_huggingface"

def read_model_from_json(dir, target_arch_family):
    with open(dir) as f:
        data = json.load(f)
    l = data[target_arch_family]
    return l

def auto_vectorize_nlp(model_name_list, output_dir_l, output_dir_p, output_dir_pl):
    patch()
    d_l, d_p, d_pl = dict(), dict(), dict()
    for n in model_name_list:
        print('Working on vectorizing', n)
        try:
            t = AutoTokenizer.from_pretrained(n, load_from_cache_file=False)
            m = AutoModel.from_pretrained(n, load_from_cache_file=False)
        except Exception as e:
            print('Failed to vectorize:', n)
            print(e)
            continue
        inp = t("Test Input", return_tensors="pt")
        gen = OrderedListGenerator(m, inp)
        fvl, fvp, fvpl = gen.vectorize()
        d_l[n] = fvl
        d_p[n] = fvp
        d_pl[n] = fvpl
        with open(output_dir_l, 'w') as f:
            json.dump(d_l, f)
        with open(output_dir_p, 'w') as f:
            json.dump(d_p, f)
        with open(output_dir_pl, 'w') as f:
            json.dump(d_pl, f)
        print('Finished on vectorizing', n)

l = read_model_from_json('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/temp_models.json', "BertForMaskedLM")
auto_vectorize_nlp(l, '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_l.json', '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_p.json', '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/test_pl.json')