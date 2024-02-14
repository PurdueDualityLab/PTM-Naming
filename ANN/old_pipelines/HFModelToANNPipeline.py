from ANN.pipelines.ANNToJSONConverter import annlayer_list_to_json
from ANN.ann_generator import AbstractNNGenerator
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor
import json
import os
import time
import sys
from PIL import Image

# The argument of setrecursionlimit is an integer that specifies the maximum number of recursive calls that can be made
sys.setrecursionlimit(5000)

def download_and_convert_model_to_json(models_dict_json_dir, output_dir, d):
    image = Image.open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/000000039769.jpg')
    with open(models_dict_json_dir) as f:
        models_dict = json.load(f)
    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/failed_models_manual.json') as f:
        failed_model_manual = json.load(f)
    counter = 0
    for model_arch in models_dict.keys():
        for model_name in models_dict[model_arch]:
            counter += 1
    s_counter = 0
    for model_arch in models_dict.keys():
        for model_name in models_dict[model_arch]:
            with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/failed_models_reason.json') as f:
                failed_model = json.load(f)
            print('\n[{}/{}] Converting'.format(s_counter, counter), model_name, '==>\n')
            s_counter += 1
            if model_name == None or model_name in failed_model.keys() or model_name in failed_model_manual:
                if model_name in failed_model_manual:
                    print('Manually skipped', model_name)
                else:
                    print('failed model:', model_name, 'due to', failed_model[model_name])
                continue
            modified_model_name = ''
            failed_model[model_name] = [None, None]
            for ch in model_name:
                if ch == '/':
                    nch = '>'
                else:
                    nch = ch
                modified_model_name += nch
            if os.path.exists(output_dir + '/' + modified_model_name + '.json'):
                print(output_dir + '/' + modified_model_name + '.json', 'exists, skipped.')
                continue

            emsg0, emsg1, emsg2, emsg3, emsg4 = None, None, None, None, None
            fc = 0
            try:
                m = AutoModel.from_pretrained(model_name, cache_dir = d)
                print('Model Generated')
            except Exception as e:
                print(e)
                emsg4 = str(e)
            try:
                t0 = AutoProcessor.from_pretrained(model_name, cache_dir = d)
                print('Process Successsful')
                inp0 = t0("Test Input", return_tensors="pt")
            except Exception as e:
                print(e)
                fc += 1
                emsg0 = str(e)
            try:
                t1 = AutoTokenizer.from_pretrained(model_name, cache_dir = d)
                print('Tokenizing Successsful')
                inp1 = t1.encode("Test Input", return_tensors="pt")
            except Exception as e:
                print(e)
                fc += 1
                emsg1 = str(e)
            try:
                t2 = AutoImageProcessor.from_pretrained(model_name, cache_dir = d)
                print('ImageProcessor Successsful')
                inp2 = t2(images=image, return_tensors="pt")
            except Exception as e:
                print(e)
                fc += 1
                emsg2 = str(e)
            try:
                t3 = AutoFeatureExtractor.from_pretrained(model_name, cache_dir = d)
                print('FeatureExtractor Successsful')
                inp3 = t3(images=image, return_tensors="pt")
            except Exception as e:
                print(e)
                fc += 1
                emsg3 = str(e)
            if fc == 4:
                print('failed to load', model_name)
                if "Too Many Requests" not in emsg0 and "Too Many Requests" not in emsg1 and "Too Many Requests" not in emsg2 and "Too Many Requests" not in emsg3: 
                    failed_model[model_name][0] = 'Model Loading'
                    failed_model[model_name][1] = (emsg0, emsg1, emsg2, emsg3, emsg4)
                else:
                    print('Too Many Requests Error')
                    time.sleep(10)
                with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/failed_models_reason.json', 'w') as f:
                    json.dump(failed_model, f)
                continue
            
            emsg0, emsg1, emsg2, emsg3 = None, None, None, None
            gc = 0
            try:
                gen = AbstractNNGenerator(m, inp0, use_hash=True)
                l_l, c_i = gen.get_connection()
                annlayer_list_to_json(l_l, c_i, output_dir + '/' + modified_model_name + '.json')
            except Exception as e:
                print(e)
                gc += 1
                emsg0 = str(e)
            if gc == 1:
                try:
                    gen = AbstractNNGenerator(m, inp1, use_hash=True)
                    l_l, c_i = gen.get_connection()
                    annlayer_list_to_json(l_l, c_i, output_dir + '/' + modified_model_name + '.json')
                except Exception as e:
                    print(e)
                    gc += 1
                    emsg1 = str(e)
            if gc == 2:
                try:
                    gen = AbstractNNGenerator(m, inp2, use_hash=True)
                    l_l, c_i = gen.get_connection()
                    annlayer_list_to_json(l_l, c_i, output_dir + '/' + modified_model_name + '.json')
                except Exception as e:
                    print(e)
                    gc += 1
                    emsg2 = str(e)         
            if gc == 3:
                try:
                    gen = AbstractNNGenerator(m, inp3, use_hash=True)
                    l_l, c_i = gen.get_connection()
                    annlayer_list_to_json(l_l, c_i, output_dir + '/' + modified_model_name + '.json')
                except Exception as e:
                    print(e)
                    gc += 1
                    emsg3 = str(e)       
            if gc == 4:
                if "CUDA" not in emsg0 and "CUDA" not in emsg1 and "CUDA" not in emsg2 and "CUDA" not in emsg3: 
                    print('fail to generate ordered list for', model_name)
                    
                    failed_model[model_name][0] = 'Tracing or List Generating'
                    failed_model[model_name][1] = (emsg0, emsg1, emsg2, emsg3)
                    with open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_vectors/failed_models_reason.json', 'w') as f:
                        json.dump(failed_model, f)
                    print(emsg0, emsg1, emsg2, emsg3)
                else:
                    print('CUDA Error')
                    del failed_model[model_name]

if __name__ == "__main__":

    download_and_convert_model_to_json(
        '/depot/davisjam/data/chingwo/PTM-Naming/model_collection/filtered_models.json',
        '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/ptm_data',
        "/scratch/gilbreth/cheung59/cache_huggingface"
        )