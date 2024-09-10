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
    with open("data_files/sql/filter_2.sql", "r", encoding="utf-8") as f:   # filter arch with >5 models, filter models with >5 downloads
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    model_list_raw = c.fetchall()
    conn.close()
    
    arch_to_model = {}
    model_to_download = {}
    for (repo_name, arch_name, download_count) in model_list_raw:
        ############################
        if "bloom" in arch_name.lower():
            logger.warning(f"Skipping {repo_name} because it is a bloom model.")
            continue
        if arch_name not in arch_to_model:
            arch_to_model[arch_name] = []
            
        arch_to_model[arch_name].append(repo_name)
        model_to_download[repo_name] = download_count
    
    
    
    selected_repos_list = []
    selected_arch_to_model = {}
    count = 0
    max_repo_count = 30 # considering some export_vec will fail (target=20)
    for arch_name, repo_list in arch_to_model.items():
        repo_list_g20 = [r for r in repo_list if model_to_download[r] > 20]
        repo_list_l20 = [r for r in repo_list if model_to_download[r] <= 20]

        if len(repo_list_g20) >= max_repo_count:
            # random sample among models with > 20 downloads
            selected_repos = random.sample(repo_list_g20, max_repo_count)
        else:
            # add all models with > 20 downloads
            selected_repos = repo_list_g20
            # random sample rest (< 20 downloads) as much as possible
            additional_repos = random.sample(repo_list_l20, min((max_repo_count - len(selected_repos)), len(repo_list_l20)))
            selected_repos.extend(additional_repos)
        if len(selected_repos) < 4:
            continue
        selected_repos_list.extend(selected_repos)
        selected_arch_to_model[arch_name] = selected_repos
        print(f"{arch_name}: {len(selected_repos)}")
        # print(f"arch_to_model: {arch_to_model[arch_name]}")
        count+=1
    print(f"architecture count: {count}")
    print(f"total dataset size: {len(selected_repos_list)}")
    # print(f"selected_arch_to_model: {selected_arch_to_model}")
    # print(f"selected_arch_to_modelt: {selected_arch_to_model}")
    

    # print(sorted_archs)
    # print(sorted_archs_l100)
    # print(len(selected_ge50_arch))
    # print(len(extra_l50_arch))
    
    # arch_to_model_ge50_keys = list(arch_to_model_ge50.keys())
    # selected_ge50_arch = random.sample(arch_to_model_ge50_keys, 50)
    # arch_to_model = arch_to_model_l50
    
    # for arch_name in selected_ge50_arch:
    #     repo_list = arch_to_model_ge50[arch_name]
    #     try:
    #         selected_repos = random.sample(repo_list, 100)
    #     except ValueError:
    #         print(f"Skipping {arch_name} because it has less than 100 models.")
    #         continue
    #     selected_repos_list.extend(selected_repos)
    #     arch_to_model[arch_name] = selected_repos
        
    # for arch_name in extra_l50_arch:
    #     repo_list = arch_to_model_l50[arch_name]
    #     selected_repos_list.extend(repo_list)
    
    # selected_repos_list.extend(random.sample(arch_to_model_l50_set, 50))
    # print the number of selected models
    # print(len(selected_repos_list))

    with open("data_files/PeaTMOSS_dataset/selected_peatmoss_repos.json", "w", encoding="utf-8") as f:
        json.dump(selected_repos_list, f)
    with open("data_files/PeaTMOSS_dataset/arch_to_repo_name.json", "w", encoding="utf-8") as f:
        json.dump(selected_arch_to_model, f)
        
    # for arch_name, repo_list in arch_to_model.items():
    #     print(f"{arch_name}: {len(repo_list)}")