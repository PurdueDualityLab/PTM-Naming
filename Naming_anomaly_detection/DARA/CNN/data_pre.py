import os
import json
import argparse
import requests
from tqdm import tqdm
from collections import defaultdict

import sqlite3
from dotenv import load_dotenv
import huggingface_hub as hf_hub


from loguru import logger

api = hf_hub.HfApi()
load_dotenv(".env")

# Set arg values
def arg_parser():
    parser = argparse.ArgumentParser(description="Process the data for the PTM-Naming project")
    parser.add_argument("--data_path", type=str, default="eval_peatmoss_data_path_final_6000/aptm", help="Path to the data folder")
    # parser.add_argument("--data_path", type=str, default="selected_peatmoss_vec_data_path/ann", help="Path to the data folder")
    parser.add_argument("--save_dir", type=str, default="Naming_anomaly_detection/DARA/CNN/data", help="Directory to save the processed data")
    parser.add_argument("--ann", type=bool, default=False, help="Whether the data is ANN or vectors")
    args = parser.parse_args()
    return args


def data_processing(data_path, save_dir, ann=False):

    if ann==True:
        '''Convert ANN to feature vectors first'''
        # TODO
    
    json_files = []
    data = {}
    lines = []
    distinct_layers = set(['Input', 'Output'])
    
    # with open("eval_peatmoss_data_path_final_6000/params.json", "r") as f:
    #     peatmoss_data = json.load(f)
    # for model, params in peatmoss_data.items():
    #     if params == -1:
    #         continue
    #     json_files.append(os.path.join(data_path, model + '.json'))
        
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    max_seq_length = 0
    # First pass to collect all unique keys and their maximum lengths                        
    for json_file in tqdm(json_files, desc="Collecting keys"):
        try:
            with open(json_file) as f:
                aptm = json.load(f)
                layer_count = 0
                for layer in aptm:
                    if 'operation' in layer:
                        distinct_layers.add(layer['operation'])
                        layer_count += 1
                max_seq_length = max(max_seq_length, layer_count)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    print(f"Max sequence length: {max_seq_length}")
    distinct_layers = sorted(distinct_layers)
    layer_to_index = {layer: i+1 for i, layer in enumerate(distinct_layers)}  # Start from 1, reserve 0 for padding
    
    for json_file in tqdm(json_files, desc="Processing files"):
        try:
            with open(json_file) as f:
                root, file = os.path.split(json_file)
                model_name = "/".join([root.split("/")[-1], file.removesuffix(".json")])
                aptm = json.load(f)

                data[model_name] = {}
                layers = [layer_to_index[layer['operation']] for layer in aptm if 'operation' in layer]
                data[model_name]['layers'] =  layers + [0] * (max_seq_length - len(layers))  #padding with 0
                data[model_name]['model_type'], data[model_name]['arch'], data[model_name]['task'] = get_model_arch_db(model_name)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error processing {json_file}: {e}")
            continue
    
    # with open("Naming_anomaly_detection/CL/data/vocabs.txt", "w") as f:
    #     f.write("\n".join(distinct_layers))
    # # Training corpus for pretraining tokenizer
    # Write the processed data to a file
    with open(f"{save_dir}/data.json", "w") as f:
        json.dump(data, f)
        
def get_task_list():    
    # with open("/depot/davisjam/data/mingyu/PTM-Naming/data_files/sql/get_distinct_task.sql", "r", encoding="utf-8") as f:
    #     query = f.read()
    
    # conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    # c = conn.cursor()
    # c.execute(query)
    
    # task_list = c.fetchall()
    # for t in task_list:
    #     print(t)
    # task_list = sorted(['unknown' if task[0] == None else task[0] for task in task_list])
    # task_list = ','.join(f"'{task}'" for task in task_list)
    
    # conn.close()Naming_anomaly_detection/
    # exit()
    with open("/depot/davisjam/data/mingyu/PTM-Naming/Naming_anomaly_detection/data_files/json_files/task_list.json", "r") as f:
        task_list = json.load(f)
    
    # return task_list
    formatted_tasks = ["'" + task + "'" if task is not None else "'unknown'" for task in task_list]
    return ','.join(formatted_tasks)

def get_model_arch_db(model_name):
    query = '''
    SELECT model.context_id, architecture.name, framework.name, 
        COALESCE(GROUP_CONCAT(DISTINCT CASE WHEN tag.name IN ({}) THEN tag.name END), 'unknown') AS tags
    FROM model
        LEFT OUTER JOIN model_to_architecture ON model.id = model_to_architecture.model_id
        LEFT OUTER JOIN architecture ON architecture.id = model_to_architecture.architecture_id
        LEFT OUTER JOIN model_to_framework ON model.id = model_to_framework.model_id
        LEFT OUTER JOIN framework ON model_to_framework.framework_id = framework.id
        LEFT OUTER JOIN model_to_tag ON model.id = model_to_tag.model_id
        LEFT OUTER JOIN tag ON model_to_tag.tag_id = tag.id
    WHERE model.context_id = ?
        AND framework.name NOT IN ('pytorch', 'tf', 'jax') 
    GROUP BY model.context_id
    '''.format(task_list)
    
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    # if model_name.split("/")[0] == "ann":
    if model_name.split("/")[0] == "aptm":
        model_name = model_name.split("/")[1]
    c.execute(query, (model_name,))
    try:
        model_info = list(c.fetchall()[0])
    except:
        return get_model_arch(model_name)
    # if model_info[3] == None:
    #     model_info[3] = get_model_arch(model_name)[2]
    if model_info[3] != 'unknown':
        model_info[3] = model_info[3].split(',')
    else:
        model_info[3] = ['unknown']
    return model_info[2], model_info[1], model_info[3]

def get_model_arch(model_name):
    '''Fetch model_type and architecture from config.json in the model's Hugging Face repository'''
    config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"  # URL to config.json
    try:
        model_info = api.model_info(model_name)
        if isinstance(model_info.pipeline_tag, str):
            task = [model_info.pipeline_tag]  # Put single task in a list
        elif isinstance(model_info.pipeline_tag, list):
            task = model_info.pipeline_tag
        else:
            task = ['unknown']
        # task = model_info.pipeline_tag if model_info.pipeline_tag else 'unknown'
    except:
        logger.warning(f"Error retrieving model info for {model_name}. Skipping...")
        return None, None, ['unknown']
    
    try:
        response = requests.get(config_url)
        response.raise_for_status()  # Raise an error for bad responses
        config = response.json()
        model_type = config.get('model_type', 'unknown')  # Replace 'unknown' with None or a default value as needed
        if len(config.get('architectures', ['unknown'])) > 1:
            print(f"{model_name} has multiple architectures: {config.get('architectures', ['unknown'])}")
        architecture = config.get('architectures', ['unknown'])[0]  # This assumes 'architectures' is a list; adjust if not
        return model_type, architecture, task
    except Exception as e:
        logger.error(f"Error retrieving model configuration: {e}")
        return None, None, task # Return None or default values for both attributes


def data_cleaning(save_dir):
    '''Remove the None architecture models from the data.json file'''
    with open(f"{save_dir}/data.json", "r") as f:
        data = json.load(f)
    failed_repos = {}
    # with open ("Naming_anomaly_detection/data_files/json_files/failed_repos.json", "r") as f:
    #     failed_repos = json.load(f)
    for model in list(data.keys()):
        if data[model]['arch'] is None or data[model]['arch'] == 'unknown':
            del data[model]
            failed_repos[model] = "Unspecified architecture"
            print(f"{model} has unspecified architecture")
        elif data[model]['task'] == ['unknown']:
            del data[model]
            failed_repos[model] = "Unspecified task"
            print(f"{model} has unspecified task")
    
    # with open ("Naming_anomaly_detection/data_files/json_files/failed_repos.json", "w") as f:
    #     json.dump(failed_repos, f)
    with open(f"{save_dir}/data_cleaned.json", "w") as f:
        json.dump(data, f)

task_list = get_task_list()

if __name__ == "__main__":
    args = arg_parser()
    data_path = args.data_path
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ann = args.ann
    data_processing(data_path, save_dir, ann=False)
    logger.success("Data processing complete.")
    data_cleaning(save_dir)
    logger.success("Data cleaning complete.")