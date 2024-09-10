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
    parser.add_argument("--data_path", type=str, default="/depot/davisjam/data/mingyu/PTM-Naming/selected_peatmoss_vec_data_path/vector", help="Path to the data folder")
    parser.add_argument("--ann", type=bool, default=False, help="Whether the data is ANN or vectors")
    args = parser.parse_args()
    return args


def data_processing(data_path="/depot/davisjam/data/mingyu/PTM-Naming/selected_peatmoss_vec_data_path/vector", ann=False):

    if ann==True:
        '''Convert ANN to feature vectors first'''
        # TODO
    
    json_files = []
    data = {}
    all_keys = defaultdict(set)  # Store all unique keys for each vector type
    
    with open("/depot/davisjam/data/mingyu/PTM-Naming/selected_peatmoss_vec_data_path/balanced_peatmoss_data.json", "r") as f:
        balanced_peatmoss_data = json.load(f)
    for model in balanced_peatmoss_data:
        repo_name = model['repo_name'] + '.json'
        json_files.append(os.path.join(data_path, repo_name))

    # open all folders under data_path and read the json files
    # Collect all json files first to provide a progress bar
    # for root, dirs, files in os.walk(data_path):
    #     for file in files:
    #         if file.endswith(".json"):
    #             json_files.append(os.path.join(root, file))

    # First pass to collect all unique keys and their maximum lengths
    for json_file in tqdm(json_files, desc="Collecting keys"):
        with open(json_file) as f:
            vecs = json.load(f)
            # Update unique keys for each type
            for vec_type in ['l', 'p', 'd']:
                if vec_type in vecs:
                    for key in vecs[vec_type].keys():
                        all_keys[vec_type].add(key)

    # Convert sets to lists and sort to maintain consistent order
    for vec_type in all_keys:
        all_keys[vec_type] = sorted(list(all_keys[vec_type]))

    # Second pass to process and save the data
    for json_file in tqdm(json_files, desc="Processing files"):
        with open(json_file) as f:
            root, file = os.path.split(json_file)
            model_name = "/".join([root.split("/")[-1], file.removesuffix(".json")])
            vecs = json.load(f)
            processed_vecs = {}
            # Process each type of vector
            for vec_type in ['l', 'p', 'd']:
                vec_data = vecs.get(vec_type, {})
                processed_vec = []
                # Convert each key in the specific vector to its index in the all_keys enumeration
                for key in all_keys[vec_type]:
                    processed_vec.append(vec_data.get(key, 0))  # Use 0 if key is not present
                processed_vecs[vec_type] = processed_vec
            
            data[model_name] = processed_vecs
            data[model_name]['model_type'], data[model_name]['arch'], data[model_name]['task'] = get_model_arch_db(model_name)

    # Write the processed data to a file
    with open("data.json", "w") as f:
        json.dump(data, f)
        
def get_task_list():    
    with open("/depot/davisjam/data/mingyu/PTM-Naming/data_files/sql/get_distinct_task.sql", "r", encoding="utf-8") as f:
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    
    task_list = c.fetchall()
    task_list = sorted(['unknown' if task[0] == None else task[0] for task in task_list])
    task_list = ','.join(f"'{task}'" for task in task_list)
    
    conn.close()
    
    return task_list

def get_model_arch_db(model_name):
    query = '''
    SELECT model.context_id, architecture.name, framework.name, 
        CASE 
            WHEN COUNT(tag.name) > 0 THEN 
                MAX(CASE WHEN tag.name IN ({}) THEN tag.name END)
            ELSE 'unknown'
        END AS tags
    FROM model
        LEFT OUTER JOIN model_to_architecture ON model.id = model_to_architecture.model_id
        LEFT OUTER JOIN architecture ON architecture.id = model_to_architecture.architecture_id
        LEFT OUTER JOIN model_to_framework ON model.id = model_to_framework.model_id
        LEFT OUTER JOIN framework ON model_to_framework.framework_id = framework.id
        LEFT OUTER JOIN model_to_tag ON model.id = model_to_tag.model_id
        LEFT OUTER JOIN tag ON model_to_tag.tag_id = tag.id
    WHERE model.context_id = ?
        AND framework.name NOT IN ('pytorch', 'tf', 'jax') 
        AND architecture.name NOT IN ('LLaMAForCausalLM') 
    GROUP BY model.context_id
    '''.format(task_list)
    
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    if model_name.split("/")[0] == "vector":
        model_name = model_name.split("/")[1]
    c.execute(query, (model_name,))
    try:
        model_info = list(c.fetchall()[0])
    except:
        return get_model_arch(model_name)
    if model_info[3] == None:
        model_info[3] = get_model_arch(model_name)[2]
        
    return model_info[2], model_info[1], model_info[3]

def get_model_arch(model_name):
    '''Fetch model_type and architecture from config.json in the model's Hugging Face repository'''
    config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"  # URL to config.json
    try:
        model_info = api.model_info(model_name)
        task = model_info.pipeline_tag if model_info.pipeline_tag else 'unknown'
    except:
        logger.warning(f"Error retrieving model info for {model_name}. Skipping...")
        return None, None, 'unknown'
    
    try:
        response = requests.get(config_url)
        response.raise_for_status()  # Raise an error for bad responses
        config = response.json()
        model_type = config.get('model_type', 'unknown')  # Replace 'unknown' with None or a default value as needed
        architecture = config.get('architectures', ['unknown'])[0]  # This assumes 'architectures' is a list; adjust if not
        return model_type, architecture, task
    except requests.exceptions.RequestException as e:
        logger.error(f"Error retrieving model configuration: {e}")
        return None, None, task # Return None or default values for both attributes


def data_cleaning():
    '''Remove the None architecture models from the data.json file'''
    with open("data.json", "r") as f:
        data = json.load(f)
    for model in list(data.keys()):
        if data[model]['arch'] is None:
            del data[model]
    with open("data_cleaned.json", "w") as f:
        json.dump(data, f)

task_list = get_task_list()

if __name__ == "__main__":
    args = arg_parser()
    data_path = args.data_path
    ann = args.ann
    data_processing(data_path, ann=False)
    logger.success("Data processing complete.")
    data_cleaning()
    logger.success("Data cleaning complete.")