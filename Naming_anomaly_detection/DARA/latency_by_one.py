import os
import traceback
import subprocess
from dotenv import load_dotenv
import json
import time
import argparse

load_dotenv()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import BitsAndBytesConfig
from loguru import logger
import matplotlib.pyplot as plt

from modeling import DARA_classifier
from dataloader import DARA_dataset
from APTM.abstract_neural_network import AbstractNN
from data_pre import get_model_arch_db
'''
load model
convert raw model weights -> APTM
evaluate
calculate throughput, latency for each model, mean std for all
'''
import sys
sys.setrecursionlimit(2000) 
# Check for GPU availability

os.environ['HF_HOME'] = os.getenv("HF_HOME")
BASE_DIR = '/depot/davisjam/data/mingyu/PTM-Naming'
PEATMOSS_PATH = os.getenv("PEATMOSS_VEC_DATA_PATH")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FAILED_REPO_PATH = "Naming_anomaly_detection/data_files/json_files/failed_final.json"

def save_failed_model(repo_name, error_message):
    """Logs the failed model with its traceback."""
    os.makedirs(os.path.dirname(FAILED_REPO_PATH), exist_ok=True)
    
    if os.path.exists(FAILED_REPO_PATH):
        with open(FAILED_REPO_PATH, "r", encoding="utf-8") as f:
            failed_models = json.load(f)
    else:
        failed_models = {}

    failed_models[repo_name] = error_message
    
    with open(FAILED_REPO_PATH, "w", encoding="utf-8") as f:
        json.dump(failed_models, f, indent=4)
    
def export_vector(repo_name):
    if not PEATMOSS_PATH:
        raise ValueError("PEATMOSS_VEC_DATA_PATH environment variable is not set")
    # if repo_name == "breadlicker45/MuseRWKV":
    #     return
    # Create base directory if it doesn't exist
    os.makedirs(PEATMOSS_PATH, exist_ok=True)
    json_file_loc_aptm = os.path.join(PEATMOSS_PATH, "aptm")
    json_file_loc_vec = os.path.join(PEATMOSS_PATH, "vector")
    os.makedirs(json_file_loc_aptm, exist_ok=True)
    os.makedirs(json_file_loc_vec, exist_ok=True)
    
    # latency json for each model
    aptm_output_path = os.path.join(json_file_loc_aptm, f"{repo_name}.json")
    vector_output_path = os.path.join(json_file_loc_vec, f"{repo_name}.json")
    

    first_letter = repo_name[0].upper()
    # load_in_4_bit = True
    model_path = os.path.join(os.getenv("LOCAL_WEIGHT_PATH", ""), first_letter, repo_name)
    logger.info(f"Model path: {model_path}")
    try:
        if os.path.exists(model_path):
            aptm, aptm_time, T_loading, total_params = AbstractNN.from_huggingface(model_path)
        else:
            aptm, aptm_time, T_loading, total_params = AbstractNN.from_huggingface(repo_name, cache_dir=os.getenv("HF_HOME"))
    
        start_time = time.time()
        with torch.no_grad():
            exclude_start_time = time.time()
            aptm.export_aptm(aptm_output_path)
            T_exclude = time.time() - exclude_start_time
            aptm.export_vector(vector_output_path)
        end_time = time.time()

        logger.success("Exported vector.")
        return vector_output_path, aptm_time, end_time - start_time - T_exclude, T_loading, total_params, T_exclude
    
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error exporting APTM for {repo_name}: {e}")
        save_failed_model(repo_name, error_message)
        return None
    

def data_processing(data_path, label_type):
    with open('Naming_anomaly_detection/data_files/json_files/all_keys.json', 'r') as f:
        all_keys = json.load(f)
    # logger.info(f"Processing data from {data_path}")
    data = {}
    with open(data_path, 'r') as f:
        root, file = os.path.split(data_path)
        model_name = "/".join([root.split("/")[-1], file.removesuffix(".json")])
        # logger.info(f"Processing model {model_name}")
        vecs = json.load(f)
        processed_vecs = {}
        # Process each type of vector
        for vec_type in ['l', 'p', 'd']:
            vec_data = vecs.get(vec_type, {})
            processed_vec = []
            # Convert each key in the specific vector to its index in the all_keys enumeration
            for key in all_keys[vec_type]:
                processed_vec.append(vec_data.get(key, 0))
            processed_vecs[vec_type] = processed_vec
        # data[model_name] = processed_vecs
        l_tensor = torch.tensor(processed_vecs['l'], dtype=torch.float).unsqueeze(0) 
        p_tensor = (1 * torch.tensor(processed_vecs['p'], dtype=torch.float)).unsqueeze(0)
        vec = torch.cat((l_tensor, p_tensor), dim=1)  # Concatenate along the new dimension

    return vec
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate a PTM, latency and FLOPs.")
    parser.add_argument("--repo", type=str, required=True, help="Repository name for the model (e.g., 'hf-internal-testing/tiny-random-LEDForConditionalGeneration')")
    parser.add_argument("--label_type", type=str, required=True, help="label type for evaluation (e.g., 'model_type', 'task', 'arch')")
    parser.add_argument("--fold", type=int, required=True, help="fold number for the model")
    args = parser.parse_args()
    
    try:
        vec_path = './data_cleaned_full_arch.json'
        full_dataset = DARA_dataset(dict_path=vec_path, label_type=args.label_type)

        index_to_label = full_dataset.get_label_mapping()
        num_classes = full_dataset.get_num_classes()
        input_shape = full_dataset.get_data_shape()
        
        json_file_loc_latency = os.path.join(PEATMOSS_PATH, "latency.json")
        os.makedirs(os.path.dirname(json_file_loc_latency), exist_ok=True)
        
        json_file_loc_flop = os.path.join(PEATMOSS_PATH, "flop.json")
        os.makedirs(os.path.dirname(json_file_loc_flop), exist_ok=True)

        if os.path.exists(json_file_loc_latency):
            with open(json_file_loc_latency, 'r') as f:
                latency = json.load(f)
        else:
            latency = {}
        if args.repo in latency and latency[args.repo] != -1:
            logger.info(f"Latency for {args.repo} already exists.")
            exit()
        # create vector
        if os.path.exists(json_file_loc_flop):
            with open(json_file_loc_flop, 'r') as f:
                params = json.load(f)
        else:
            params = {}
        
        logger.info(f"Exporting APTM for {args.repo}")
        sample_start_time = time.time()
        result = export_vector(args.repo)
            
        # failed to export vector
        latency[args.repo] = -1
        if not result:
            with open(json_file_loc_latency, 'w') as f:
                json.dump(latency, f)
        output_path, T_aptm, T_export, T_loading, total_params, T_exclude = result
        
        exclude_start_time = time.time()
        params[args.repo] = total_params
        with open(json_file_loc_flop, "w") as f:
            json.dump(params, f)
        logger.success(f"Total # of Params for {args.repo} calculated and saved.")
        T_exclude += time.time() - exclude_start_time
            
        # data processing
        process_start_time = time.time()
        data = data_processing(data_path=output_path, label_type=args.label_type)
        T_process = time.time() - process_start_time
        
        # Initialize the model for the current fold
        prediction_start_time = time.time()
        model = DARA_classifier(input_size=input_shape[1], output_size=num_classes)
        PATH = f'{BASE_DIR}/fold_{args.fold}_model_state_dict_final.pt'
        model.load_state_dict(torch.load(PATH))
        model = model.to(device)
        model.eval()
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        T_prediction = time.time() - prediction_start_time
        
        # logger.info(f'index_to_label: {index_to_label}, predict: {predicted.item()}, target: {target[i]}')
        logger.info(f'predict: {index_to_label[predicted.item()]}') #, target: {args.index_to_label[target[i].item()]}')
        T_total = time.time() - sample_start_time - T_exclude
        T_overhead = T_total - (T_aptm + T_export + T_process + T_prediction + T_loading)
        # latency calculation
        latency[args.repo] = {
            'aptm': T_aptm,
            'export': T_export,
            'process': T_process,
            'prediction': T_prediction,
            'loading': T_loading,
            'overhead': T_overhead,
            'total': T_total
        }
        with open(json_file_loc_latency, 'w') as f:
            json.dump(latency, f)
        logger.success(f"Latency for {args.repo} calculated and saved.")
        
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error exporting APTM for {args.repo}: {e}")
        save_failed_model(args.repo, error_message)
        return None

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
    # repo_name = "PlanTL-GOB-ES/longformer-base-4096-bne-es"
    # repo_name = "mafwalter/roberta-base-finetuned-question-v-statement"
    # repo_name = "kazzand/ru-longformer-tiny-16384"
    # repo_name = "cloudqi/cqi_speech_recognize_pt_v0"
    # CV_run()
    # repo_name = "beomi/KoRWKV-6B"
    # repo_name = "Salesforce/codegen2-16B"
    # repo_name = "breadlicker45/MuseRWKV"
    # repo_name = "hfl/chinese-xlnet-base"
    # repo_name = "hf-internal-testing/tiny-random-LEDForConditionalGeneration"
    # Tracing input: {'input_ids': tensor([[    0, 34603, 41327,     2]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1]], device='cuda:0')}, type: <class 'transformers.tokenization_utils_base.BatchEncoding'>
    # Tracing input: {'input_ids': tensor([[   0,  106, 6378,  657,  714,  138,    2]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}, type: <class 'transformers.tokenization_utils_base.BatchEncoding'>