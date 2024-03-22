"""
This script is used to load the peatmoss data into JSON files.
"""
import sqlite3
import os
import random
import json
from dotenv import load_dotenv
from loguru import logger

if __name__ == "__main__":
    load_dotenv(".env")
    with open("data_files/sql/get_peatmoss_models_dn_g50.sql", "r", encoding="utf-8") as f:
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    model_list_raw = c.fetchall()
    conn.close()
    # write to JSON file
    with open("data_files/json_files/selected_peatmoss_repos_dn_g50.json", "w", encoding="utf-8") as f:
        json.dump([repo_name_tuple[0] for repo_name_tuple in model_list_raw], f, indent=4)