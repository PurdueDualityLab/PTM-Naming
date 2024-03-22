"""
This script is used to convert the random sample of peatmoss models to a vectorized format.
"""

import json
import subprocess
import time
import os
from dotenv import load_dotenv
from loguru import logger

def run_process():
    """Run the main application process."""

    load_dotenv(".env")
    module_use_path = str(os.getenv("MODULE_USE_PATH"))
    module_load_path = str(os.getenv("MODULE_LOAD_PATH"))
    python_path = str(os.getenv("PYTHONPATH"))
    if not os.path.exists(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/rand_sample"):
        os.makedirs(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/rand_sample", exist_ok=True)
    json_file_loc = f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/rand_sample"
    model_list_json_loc = "data_files/json_files/selected_peatmoss_repos.json"
    if not os.path.exists(model_list_json_loc):
        raise FileNotFoundError(f"File {model_list_json_loc} does not exist.")

    command = f"""
    module use {module_use_path} && 
    module load {module_load_path} && 
    export PYTHONPATH=$PYTHONPATH:{python_path} &&
    python data_files/scripts/export_vec.py -j {json_file_loc} -l {model_list_json_loc}
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
    idx = -1
    while idx < 9000:
        p = run_process()
        res = monitor_process(p)
        with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
            idx = int(f.readline())
        