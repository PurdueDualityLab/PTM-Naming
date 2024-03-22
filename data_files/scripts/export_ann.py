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
from ANN.abstract_neural_network import AbstractNN

def get_ordered_model_list() -> list:
    """
    Get ordered model list.

    Returns:
        list: list of models
    """
    load_dotenv(".env")
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute("SELECT model.context_id FROM model")
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
    if len(sys.argv) != 5:
        print("Invalid number of arguments.")
        sys.exit(1)
    elif sys.argv[1] == "help":
        print("This script exports ann from a hf repository.")
        print("Usage: python export_ann.py -j <json_output_loc> -c <run_count>")
        sys.exit(0)
    elif sys.argv[1] != "-j" or sys.argv[3] != "-c":
        print("Invalid arguments.")
        sys.exit(1)
    else:
        json_output_loc = sys.argv[2]
        run_count = int(sys.argv[4])

    model_list = get_ordered_model_list()
    # model_list = get_local_model_list()
    torchview.torchview.forward_prop = forward_prop
    with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
        idx = int(f.readline())
    logger.info(f"Index: {idx}.")
    model_list = model_list[idx:]
    count = 0
    for i, repo_name in enumerate(model_list):

        if count >= run_count:
            break

        start_time = time.time()
        idx += 1
        count += 1
        with open("data_files/other_files/temp_index.txt", "w", encoding="utf-8") as f:
            f.seek(0)
            f.writelines(str(idx))
        logger.info(f"[{i}] Processing {repo_name}.")
        try:
            if os.path.exists(json_output_loc + f"/{repo_name}.json"):
                logger.sucesss(f"File {json_output_loc + f'/{repo_name}'}.json already exists.")
                logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                continue
            
            if os.path.exists(str(os.getenv("LOCAL_WEIGHT_PATH")) + '/' + repo_name):
                try:
                    logger.debug(f"Loading {repo_name} in_4bit=True")
                    ann = AbstractNN.from_huggingface(
                        str(os.getenv("LOCAL_WEIGHT_PATH")) + '/' + repo_name,
                        load_in_4bit = True
                    )
                
                except:
                    logger.debug(f"Loading {repo_name} in_4bit=False")
                    ann = AbstractNN.from_huggingface(
                        str(os.getenv("LOCAL_WEIGHT_PATH")) + '/' + repo_name,
                        load_in_4bit = False
                    )

            else:
                ############
                # TODO: Remove this if need to download weights from HF
                # Just load the local files
                logger.warning(f"Model {repo_name} not found in local weights. Continueing...")
                continue
                ############
                ann = AbstractNN.from_huggingface(
                    repo_name,
                    load_in_4bit = True
                )
            curr_json_output_loc = json_output_loc + f"/{repo_name}.json"
            if not os.path.exists(curr_json_output_loc):
                os.makedirs('/'.join(curr_json_output_loc.split('/')[:-1]), exist_ok=True)
            ann.export_ann(json_output_loc + f"/{repo_name}.json")
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
        except Exception as emsg: # pylint: disable=broad-except

            if not os.path.exists("data_files/json_files/failed_ann.json"):
                with open("data_files/json_files/failed_ann.json", "w", encoding="utf-8") as f:
                    json.dump({}, f)
            with open("data_files/json_files/failed_ann.json", "r", encoding="utf-8") as f:
                failed_ann = json.load(f)

            tb_str = traceback.format_exc()
            failed_ann[repo_name] = tb_str

            with open("data_files/json_files/failed_ann.json", "w", encoding="utf-8") as f:
                json.dump(failed_ann, f)

            logger.error(tb_str)
            logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
