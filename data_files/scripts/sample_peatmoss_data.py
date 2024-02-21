"""
This script is used to load the peatmoss data into JSON files.
"""

import subprocess
import time
import sqlite3
import os
import random
from dotenv import load_dotenv
from loguru import logger

def run_process(repo_name: str, json_file_loc: str):
    """Run the main application process."""

    load_dotenv(".env")
    module_use_path = str(os.getenv("MODULE_USE_PATH"))
    module_load_path = str(os.getenv("MODULE_LOAD_PATH"))
    python_path = str(os.getenv("PYTHONPATH"))
    sql_file_loc = "data_files/sql/get_peatmoss_models.sql"

    command = f"""
    module use {module_use_path} && 
    module load {module_load_path} && 
    export PYTHONPATH=$PYTHONPATH:{python_path} &&
    python data_files/scripts/export_vec.py -r {repo_name} -j {json_file_loc} -sql {sql_file_loc}
    """

    return subprocess.Popen(command, shell=True, executable='/bin/bash')

def monitor_process(p: subprocess.Popen) -> int:
    """Monitor the main application process."""

    while True:
        result = p.poll()
        if result is not None:  # Process has exited
            logger.info(f"Process has exited with {result}.")
            return result
        time.sleep(0.1)

if __name__ == "__main__":
    load_dotenv(".env")
    with open("data_files/sql/get_arch_model_list.sql", "r", encoding="utf-8") as f:
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    model_list_raw = c.fetchall()
    conn.close()
    
    arch_to_model = {}
    for (repo_name, arch_name) in model_list_raw:
        if arch_name not in arch_to_model:
            arch_to_model[arch_name] = []
        arch_to_model[arch_name].append(repo_name)
    
    arch_to_model_ge50 = {}
    for arch_name, repo_list in arch_to_model.items():
        if len(repo_list) >= 50:
            arch_to_model_ge50[arch_name] = repo_list
    arch_to_model_l50 = {}
    for arch_name, repo_list in arch_to_model.items():
        if len(repo_list) < 50:
            arch_to_model_l50[arch_name] = repo_list

    for arch_name, repo_list in arch_to_model_ge50.items():
        pass
    '''
    with open("data_files/sql/get_peatmoss_models.sql", "r", encoding="utf-8") as f:
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    model_list_raw = c.fetchall()
    conn.close()

    model_list = [tuple_[0] for tuple_ in model_list_raw]

    for repo_name_ in model_list:
        start_time = time.time()
        json_file_loc_ = f"peatmoss_ann/{repo_name_}"
        p_ = run_process(repo_name_, json_file_loc_)
        R = monitor_process(p_)
        end_time = time.time()
        logger.info(f"Time taken for {repo_name_}: {end_time - start_time} seconds.")
    '''