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
        ############################
        if "bloom" in arch_name.lower():
            logger.warning(f"Skipping {repo_name} because it is a bloom model.")
            continue
        if arch_name not in arch_to_model:
            arch_to_model[arch_name] = []
        arch_to_model[arch_name].append(repo_name)
    
    arch_to_model_ge50 = {}
    arch_to_model_l50_set = set()
    arch_to_model_l50 = {}
    for arch_name, repo_list in arch_to_model.items():
        if len(repo_list) >= 100:
            arch_to_model_ge50[arch_name] = repo_list
        elif len(repo_list) >= 20:
            arch_to_model_l50[arch_name] = repo_list
            for repo_name in repo_list:
                arch_to_model_l50_set.add(repo_name)

    selected_repos_list = []
    # Convert dictionary items to a list of tuples (key-value pairs) and sort them based on the count (value) in descending order
    
    # print({k: len(v) for k, v in arch_to_model_ge50.items()})
    # print({k: len(v) for k, v in arch_to_model_l50.items()})
     

    sorted_archs = sorted(arch_to_model_ge50.items(), key=lambda item: item[1], reverse=True)
    sorted_archs_l100 = sorted(arch_to_model_l50.items(), key=lambda item: item[1], reverse=True)
    # Remove the key (architecture name) from the sorted list
    

    
    # Select all the architectures with over 1000 models
    # selected_1000_arch = [item[0] for item in sorted_archs if len(item[1]) >= 100]

    # # Select the top 50 architectures from the sorted list
    selected_ge50_arch = [item[0] for item in sorted_archs[:100]]
    extra_l50_arch = [item[0] for item in sorted_archs_l100]


    # print(sorted_archs)
    # print(sorted_archs_l100)
    # print(len(selected_ge50_arch))
    # print(len(extra_l50_arch))
    
    # arch_to_model_ge50_keys = list(arch_to_model_ge50.keys())
    # selected_ge50_arch = random.sample(arch_to_model_ge50_keys, 50)
    arch_to_model = arch_to_model_l50
    
    for arch_name in selected_ge50_arch:
        repo_list = arch_to_model_ge50[arch_name]
        try:
            selected_repos = random.sample(repo_list, 100)
        except ValueError:
            print(f"Skipping {arch_name} because it has less than 100 models.")
            continue
        selected_repos_list.extend(selected_repos)
        arch_to_model[arch_name] = selected_repos
        
    for arch_name in extra_l50_arch:
        repo_list = arch_to_model_l50[arch_name]
        selected_repos_list.extend(repo_list)
    
    # selected_repos_list.extend(random.sample(arch_to_model_l50_set, 50))
    # print the number of selected models
    print(len(selected_repos_list))

    with open("data_files/json_files/new_selected_peatmoss_repos.json", "w", encoding="utf-8") as f:
        json.dump(selected_repos_list, f)
    with open("data_files/json_files/temp_arch_to_model.json", "w", encoding="utf-8") as f:
        json.dump(arch_to_model, f)
        
    for arch_name, repo_list in arch_to_model.items():
        print(f"{arch_name}: {len(repo_list)}")