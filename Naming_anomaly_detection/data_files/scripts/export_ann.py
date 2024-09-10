"""
This script exports a single ann from a hf repository.
"""

import sys
import time
import json
import os
from loguru import logger
import sqlite3
import traceback
import torchview
from convert_all_peatmoss_models_to_ann import forward_prop
from dotenv import load_dotenv
from APTM.abstract_neural_network import AbstractNN
from transformers import BitsAndBytesConfig
import torch

def get_ordered_model_list() -> list:
    """
    Get ordered model list.

    Returns:
        list: list of models
    """
    load_dotenv(".env")
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute("SELECT model.context_id FROM model WHERE model.downloads >= 50")
    repo_names = c.fetchall()
    return sorted([repo_name[0] for repo_name in repo_names])


def get_local_model_list() -> list:
    """
    Get local model list.

    Returns:
        list: list of models
    """
    # load_dotenv(".env")
    # conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    # c = conn.cursor()
    # c.execute("SELECT DISTINCT model.context_id, has_snapshot FROM model WHERE has_snapshot is True;")
    # repo_names = c.fetchall()

    repo_names = []
    base_dir = os.getenv("LOCAL_WEIGHT_PATH")

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_dir):
        # Only proceed if we're not at the base directory level
        if root != base_dir:
            # Check if '.config' file exists in the directory
            if 'config.json' in files:
                # Construct "author/model_name" from the path, assuming the structure is always /base_dir/author/model_name
                author_model_name = os.path.relpath(root, base_dir)
                # Add the constructed "author/model_name" to the list
                repo_names.append(author_model_name)

    # Log the list of repository names for debugging
    # logger.debug(repo_names)

    logger.info(f"-----------------Loading {len(repo_names)} models from local path-----------------")
    # print has_snapshots
    return sorted([repo_name for repo_name in repo_names])


if __name__ == "__main__":
    if len(sys.argv) != 7 and len(sys.argv) != 9:
        logger.error("Invalid number of arguments.")
        sys.exit(1)
    elif sys.argv[1] == "help":
        logger.error("This script exports ann from a hf repository.")
        logger.error("Usage: python export_ann.py -j <json_output_loc> -c <run_count>")
        sys.exit(0)
    elif sys.argv[1] != "-ja" or sys.argv[3] != "-jv" or sys.argv[5] != "-c":
        logger.error("Invalid arguments.")
        sys.exit(1)
    else:
        json_output_loc_ann = sys.argv[2]
        json_output_loc_vec = sys.argv[4]
        json_output_loc_intermediate = sys.argv[2].split('/')[0] + '/intermediate'
        run_count = int(sys.argv[6])
        
    if sys.argv[7] != "-s": #sampled list
        model_list = get_ordered_model_list()
    else:
        with open(sys.argv[8], "r", encoding="utf-8") as f:
            model_list = json.load(f)
            model_list = list(model_list.keys())    # temp
    # model_list = get_local_model_list()
    torchview.torchview.forward_prop = forward_prop
    with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
        idx = int(f.readline())
    logger.info(f"Index: {idx}.")
    model_list = model_list[idx:]
    count = 0
    for i, repo_name in enumerate(model_list):
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if count >= run_count:
            break

        start_time = time.time()
        idx += 1
        count += 1
        with open("data_files/other_files/temp_index.txt", "w", encoding="utf-8") as f:
            f.seek(0)
            f.writelines(str(idx))
        if 'blip' in repo_name or 'Blip' in repo_name or 'clip-roberta-finetuned' in repo_name:
            logger.info(f"Skipping {repo_name}.")
            continue
        logger.info(f"[{i}] Processing {repo_name}.")
        try:
            # Check if model is quantized
            if not os.path.exists("data_files/json_files/selected_peatmoss_repos.json"):
                with open("data_files/json_files/selected_peatmoss_repos.json", "w", encoding="utf-8") as f:
                    json.dump({}, f)
            quantized_model = {}
            with open("data_files/json_files/selected_peatmoss_repos.json", "r", encoding="utf-8") as f:
                quantized_model = json.load(f)
                
            # if os.path.exists(json_output_loc_vec + f"/{repo_name}.json"):
            if os.path.exists(json_output_loc_ann + f"/{repo_name}.json") and os.path.exists(json_output_loc_vec + f"/{repo_name}.json"):# and os.path.exists(json_output_loc_intermediate + f"/{repo_name}.json"):
                logger.success(f"{repo_name} files already exist.")
                logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                continue
            
            first_letter = repo_name[0].upper()
            load_in_4_bit = True
            if os.path.exists(str(os.getenv("LOCAL_WEIGHT_PATH")) + f'/{first_letter}/' + repo_name):
                try:
                    logger.debug(f"Loading {repo_name} in_4bit=True")
                    q_config = BitsAndBytesConfig(load_in_4_bit=True)
                    ann = AbstractNN.from_huggingface(
                        str(os.getenv("LOCAL_WEIGHT_PATH")) + f'/{first_letter}/' + repo_name,
                        quantization_config=q_config,
                        # load_in_4bit = True
                    )
                except:
                    logger.debug(f"Loading {repo_name} in_4bit=False")
                    # q_config = BitsAndBytesConfig(load_in_4_bit=False)
                    ann = AbstractNN.from_huggingface(
                        str(os.getenv("LOCAL_WEIGHT_PATH")) + f'/{first_letter}/' + repo_name,
                        # quantization_config=q_config,
                        # load_in_4bit = False
                    )
                    load_in_4_bit = False
            else:
                ############
                # TODO: Remove this if need to download weights from HF
                # Just load the local files
                # logger.warning(f"Model {repo_name} not found in local weights. Continuing...")
                # continue
                ############
                logger.info("Downloading model checkpoints from HF...")
                try:
                    logger.debug(f"Loading {repo_name} in_4bit=True")
                    q_config = BitsAndBytesConfig(load_in_4_bit=True)
                    ann = AbstractNN.from_huggingface(
                        repo_name,
                        quantization_config=q_config,
                        cache_dir='/scratch/gilbreth/kim3118/.cache/huggingface'
                    )
                    load_in_4_bit = True
                except:
                    logger.debug(f"Loading {repo_name} in_4bit=False")
                    ann = AbstractNN.from_huggingface(
                        repo_name,
                        cache_dir='/scratch/gilbreth/kim3118/.cache/huggingface'
                    )
                    load_in_4_bit = False

            
            curr_json_output_loc = json_output_loc_ann + f"/{repo_name}.json"
            if not os.path.exists(curr_json_output_loc):
                os.makedirs(os.path.dirname(curr_json_output_loc), exist_ok=True)
                ann.export_ann(json_output_loc_ann + f"/{repo_name}.json")
                logger.success("Exported ann.")
            else:
                logger.info("ANN file already exists.")
            curr_json_output_loc = json_output_loc_vec + f"/{repo_name}.json"
            if not os.path.exists(curr_json_output_loc):
                # os.makedirs(os.path.dirname(curr_json_output_loc), exist_ok=True)
                ann.export_vector(curr_json_output_loc + f"/{repo_name}.json")
                logger.success("Exported vector")
            else:
                logger.info("Vector file already exists.")
            
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                
            quantized_model[repo_name] = load_in_4_bit
            with open("data_files/json_files/selected_peatmoss_repos.json", "w", encoding="utf-8") as f:
                json.dump(quantized_model, f)
            
            failed_aptms = {}
            with open("data_files/json_files/failed_aptm.json", 'r') as f:
                failed_aptms = json.load(f)
            if repo_name in failed_aptms:
                del failed_aptms[repo_name]
                with open("data_files/json_files/failed_aptm.json", 'w') as f:
                    json.dump(failed_aptms, f)
                
        except Exception as emsg: # pylint: disable=broad-except
            json_file_path = "data_files/json_files/failed_aptm.json"
            os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
            if not os.path.exists("data_files/json_files/failed_aptm.json"):
                with open("data_files/json_files/failed_aptm.json", "w", encoding="utf-8") as f:
                    json.dump({}, f)
            
            failed_ann = {}
            with open("data_files/json_files/failed_aptm.json", "r", encoding="utf-8") as f:
                failed_ann = json.load(f)
            tb_str = traceback.format_exc()
            failed_ann[repo_name] = tb_str

            with open("data_files/json_files/failed_aptm.json", "w", encoding="utf-8") as f:
                json.dump(failed_ann, f)

            logger.error(tb_str)
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
            
            if "CUDA error: device-side assert triggered" in tb_str or 'TORCH_USE_CUDA_DSA' in tb_str:
                logger.error("CUDA error: device-side assert triggered. Exiting...")
                break
