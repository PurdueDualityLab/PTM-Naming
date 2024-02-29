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
                logger.info(f"File {repo_name}.json already exists.")
                logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                continue

            # check if the repo is already a bad model, or requires remote code / authorization
            if os.path.exists("data_files/json_files/bad_models.json"):
                with open("data_files/json_files/bad_models.json", "r", encoding="utf-8") as f:
                    bad_models = json.load(f)
                if repo_name in bad_models:
                    logger.error(f"{repo_name} is a bad model.")
                    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                    continue
            if os.path.exists("data_files/json_files/requires_remote_code.json"):
                with open(
                    "data_files/json_files/requires_remote_code.json", 
                    "r", 
                    encoding="utf-8"
                ) as f:
                    requires_remote_code = json.load(f)
                if repo_name in requires_remote_code:
                    logger.error(f"{repo_name} requires remote code.")
                    logger.info(f"Time taken: {time.time() - start_time:.2f} seconds.")
                    continue
            
            if os.path.exists(str(os.getenv("LOCAL_WEIGHT_PATH")) + '/' + repo_name):
                ann = AbstractNN.from_huggingface(
                    str(os.getenv("LOCAL_WEIGHT_PATH")) + '/' + repo_name,
                    load_in_4bit = True
                )
            else:
                ann = AbstractNN.from_huggingface(
                    repo_name,
                    load_in_4bit = True
                )
            if not os.path.exists(json_output_loc + f"/{repo_name}.json"):
                os.makedirs(json_output_loc, exist_ok=True)
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
