import os
import json
import argparse
import requests
from tqdm import tqdm
from collections import defaultdict

import huggingface_hub as hf_hub


from loguru import logger

api = hf_hub.HfApi()

# Set arg values
def arg_parser():
    parser = argparse.ArgumentParser(description="Process the data for the PTM-Naming project")
    parser.add_argument("--data_path", type=str, default="/depot/davisjam/data/chingwo/PTM-v2/PTM-Naming/peatmoss_ann/rand_sample", help="Path to the data folder")
    parser.add_argument("--ann", type=bool, default=False, help="Whether the data is ANN or vectors")
    args = parser.parse_args()
    return args


def data_processing(data_path="/depot/davisjam/data/chingwo/PTM-v2/PTM-Naming/peatmoss_ann/rand_sample", ann=False):

    if ann==True:
        '''Convert ANN to feature vectors first'''
        # TODO
    
    # open all folders under data_path and read the json files
    data = {}
    all_keys = defaultdict(set)  # Store all unique keys for each vector type

    # Collect all json files first to provide a progress bar
    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

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
            data[model_name]['model_type'], data[model_name]['arch'], data[model_name]['task'] = get_model_arch(model_name)

    # Write the processed data to a file
    with open("data.json", "w") as f:
        json.dump(data, f)

def get_model_arch(model_name):
    '''Fetch model_type and architecture from config.json in the model's Hugging Face repository'''
    config_url = f"https://huggingface.co/{model_name}/raw/main/config.json"  # URL to config.json
    try:
        model_info = api.model_info(model_name)
        task = model_info.pipeline_tag if model_info.pipeline_tag else 'unknown'
    except:
        logger.warning(f"Error retrieving model info for {model_name}. Skipping...")
        return None, None, None
    
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


if __name__ == "__main__":
    args = arg_parser()
    data_path = args.data_path
    data_path = "/depot/davisjam/data/chingwo/PTM-v2/PTM-Naming/peatmoss_ann/rand_sample_2500"
    ann = args.ann
    data_processing(data_path, ann=False)
    logger.success("Data processing complete.")
    data_cleaning()
    logger.success("Data cleaning complete.")