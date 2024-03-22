"""
This script is used to load the peatmoss data (hf commit date) into JSON files.
"""
import sqlite3
import os
import random
import json
from dotenv import load_dotenv
from loguru import logger

def compare_time_string(t1, t2):
    # compare date and time string in this format: '2022-07-06 21:54:25+00:00'
    # return 1 if t1 is later than t2, 0 if t1 is the same as t2, -1 if t1 is earlier than t2
    t1 = t1.split(" ")
    t2 = t2.split(" ")
    date1 = t1[0].split("-")
    date2 = t2[0].split("-")
    time1 = t1[1].split(":")
    time2 = t2[1].split(":")
    if int(date1[0]) > int(date2[0]):
        return 1
    elif int(date1[0]) < int(date2[0]):
        return -1
    else:
        if int(date1[1]) > int(date2[1]):
            return 1
        elif int(date1[1]) < int(date2[1]):
            return -1
        else:
            if int(date1[2]) > int(date2[2]):
                return 1
            elif int(date1[2]) < int(date2[2]):
                return -1
            else:
                if int(time1[0]) > int(time2[0]):
                    return 1
                elif int(time1[0]) < int(time2[0]):
                    return -1
                else:
                    if int(time1[1]) > int(time2[1]):
                        return 1
                    elif int(time1[1]) < int(time2[1]):
                        return -1
                    else:
                        if int(time1[2].split("+")[0]) > int(time2[2].split("+")[0]):
                            return 1
                        elif int(time1[2].split("+")[0]) < int(time2[2].split("+")[0]):
                            return -1
                        else:
                            return 0

if __name__ == "__main__":
    load_dotenv(".env")
    with open("data_files/sql/get_model_hf_commit_data.sql", "r", encoding="utf-8") as f:
        query = f.read()
    conn = sqlite3.connect(str(os.getenv("PEATMOSS_DB")))
    c = conn.cursor()
    c.execute(query)
    model_list_raw = c.fetchall()
    conn.close()
    # for each tuple (model_name, commit_time), convert to a dictionary of
    # model_name: earliest_commit_time
    model_dict = {}
    for model_name, commit_time in model_list_raw:
        if model_name not in model_dict:
            model_dict[model_name] = commit_time
        else:
            if compare_time_string(commit_time, model_dict[model_name]) == -1:
                model_dict[model_name] = commit_time
    # write to JSON file
    with open("data_files/json_files/model_hf_commit_data.json", "w", encoding="utf-8") as f:
        json.dump(model_dict, f, indent=4)