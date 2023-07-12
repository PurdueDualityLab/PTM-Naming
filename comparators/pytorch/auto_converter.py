from list_to_json import node_list_to_json
from list_gen import OrderedListGenerator
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor
import json
import os

def download_and_convert_model_to_json(models_dict_json_dir, output_dir, d):
    with open(models_dict_json_dir) as f:
        models_dict = json.load(f)
    for model_arch in models_dict.keys():
        for model_name in models_dict[model_arch]:
            if model_name == None:
                continue
            modified_model_name = ''
            for ch in model_name:
                if ch == '/':
                    nch = '>'
                else:
                    nch = ch
                modified_model_name += nch
            if os.path.exists(output_dir + '/' + modified_model_name + '.json'):
                print(output_dir + '/' + modified_model_name + '.json', 'exists, skipped.')
                continue
            fc = 0
            try:
                t = AutoProcessor.from_pretrained(model_name, cache_dir = d)
                m = AutoModel.from_pretrained(model_name, cache_dir = d)
                inp = t("Test Input", return_tensors="pt")
            except Exception as e:
                print(e)
                fc += 1
            if fc == 1:
                print('failed to load', model_name)
                continue
            
            try:
                gen = OrderedListGenerator(m, inp, use_hash=True)
                l_l, c_i = gen.get_connection()
                node_list_to_json(l_l, c_i, output_dir + '/' + modified_model_name + '.json')
            except:
                print('fail to generate ordered list for', model_name)

download_and_convert_model_to_json(
    '/depot/davisjam/data/chingwo/PTM-Naming/model_collection/filtered_models.json',
    '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_data',
    "/scratch/gilbreth/cheung59/cache_huggingface"
    )

# end at bigbird-pegasus-large-bigpatent.json