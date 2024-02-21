"""
This script is used to load the peatmoss data into JSON files.
"""

import subprocess
import time
import sqlite3
import os
import random
import json
from dotenv import load_dotenv
from loguru import logger

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
    arch_to_model_l50_set = set()
    for arch_name, repo_list in arch_to_model.items():
        if len(repo_list) < 50:
            for repo_name in repo_list:
                arch_to_model_l50_set.add(repo_name)

    selected_repos_list = []

    arch_to_model_ge50_keys = list(arch_to_model_ge50.keys())
    selected_ge50_arch = random.sample(arch_to_model_ge50_keys, 19)
    for arch_name in selected_ge50_arch:
        repo_list = arch_to_model_ge50[arch_name]
        selected_repos = random.sample(repo_list, 50)
        selected_repos_list.extend(selected_repos)
    
    selected_repos_list.extend(random.sample(arch_to_model_l50_set, 50))

    with open("data_files/json_files/selected_peatmoss_repos.json", "w", encoding="utf-8") as f:
        json.dump(selected_repos_list, f)
    