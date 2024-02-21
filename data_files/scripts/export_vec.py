"""
This script exports a single vector from a hf repository.
"""

import json
import os
from pyexpat import model
import sys
import sqlite3
from loguru import logger
from dotenv import load_dotenv
from ANN.abstract_neural_network import AbstractNN

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Invalid number of arguments.")
        sys.exit(1)
    elif sys.argv[1] == "help":
        print("This script exports vectors from a hf repository.")
        print("Usage: python export_vec.py -j <json_output_loc> -l <model_list_json_loc>")
        sys.exit(0)
    elif sys.argv[1] != "-j" or sys.argv[3] != "-l":
        print("Invalid arguments.")
        sys.exit(1)
    else:
        json_output_loc = sys.argv[2]
        model_list_json_loc = sys.argv[4]
    with open(model_list_json_loc, "r", encoding="utf-8") as f:
        model_list = json.load(f)
    with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
        idx = int(f.readline())
    logger.info(f"Index: {idx}.")
    model_list = model_list[idx:]
    for i, repo_name in enumerate(model_list):
        idx += 1
        with open("data_files/other_files/temp_index.txt", "w", encoding="utf-8") as f:
            f.seek(0)
            f.writelines(str(idx))
        logger.info(f"[{i}] Processing {repo_name}.")
        try:
            ann = AbstractNN.from_huggingface(repo_name)
            if os.path.exists(json_output_loc + f"/{repo_name}.json"):
                logger.info(f"File {repo_name}.json already exists.")
                continue
            else:
                os.makedirs(json_output_loc, exist_ok=True)
            ann.export_vector(json_output_loc + f"/{repo_name}.json")
        except Exception as emsg: # pylint: disable=broad-except
            logger.error(emsg)
            sys.exit(1)
