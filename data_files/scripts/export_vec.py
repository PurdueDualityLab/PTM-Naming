"""
This script exports a single vector from a hf repository.
"""

import os
from pyexpat import model
import sys
import sqlite3
from loguru import logger
from dotenv import load_dotenv
from ANN.abstract_neural_network import AbstractNN

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Invalid number of arguments.")
        sys.exit(1)
    elif sys.argv[1] == "help":
        print("This script exports vectors from a hf repository.")
        print("Usage: python export_vec.py -r <repo_name> -j <json_output_loc> -l <model_list_json_loc>")
        sys.exit(0)
    elif sys.argv[1] != "-r":
        print("Invalid argument.")
        sys.exit(1)
    elif sys.argv[3] != "-j":
        print("Invalid argument.")
        sys.exit(1)
    elif sys.argv[5] != "-l":
        print("Invalid argument.")
        sys.exit(1)
    else:
        repo_name = sys.argv[2]
        json_output_loc = sys.argv[4]
        model_list_json_loc = sys.argv[6]
    with open(model_list_json_loc, "r", encoding="utf-8") as f:
        model_list = f.read()
    with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
        idx = f.readline()
    logger.info(f"Index: {idx}.")
    model_list = model_list[int(idx):]
    for i, repo_name in enumerate(model_list):
        with open("data_files/other_files/temp_index.txt", "w", encoding="utf-8") as f:
            f.seek(0)
            f.writelines(str(i))
        logger.info(f"[{i}] Processing {repo_name}.")
        try:
            ann = AbstractNN.from_huggingface(repo_name)
            ann.export_vector(json_output_loc)
        except Exception as emsg: # pylint: disable=broad-except
            logger.error(emsg)
            sys.exit(1)
