"""
This file contains the functions to get the GPT response.
"""

import os
import json
import random
import time
import traceback
from dotenv import load_dotenv
from openai import OpenAI
from loguru import logger
from name_analysis import prompt

CATEGORIES = ['A', 'S', 'D', 'C', 'V', 'F', 'L', 'T', 'R', 'N', 'H', 'P', 'O']

def chat(question_content: str):
    """
    This function sends a chat request to the GPT model.

    Args:
        question_content: The question to ask the GPT model.

    Returns:
        response: The response from the GPT model.
    """
    load_dotenv(".env")
    chatlog = []
    chatlog.append({"role" : "system", "content" : prompt.BACKGROUND})
    chatlog.append({"role" : "user", "content" : question_content})
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=chatlog,
        temperature=0.3,
        logprobs=True
    )
    return response

def get_category_list(response: str):
    """
    This function gets the category list from the response.

    Args:
        response: The response from the GPT model.

    Returns:
        cat_list: The list of categories.
    """
    # split the string by :
    [_, cat_set] = response.split(": ")
    # split the category by ,
    cat_list = cat_set.split(", ")
    # remove the redundant category
    cat_list = list(set(cat_list))
    return cat_list

def run_part(start: int, end: int):
    """
    This function runs part the model analysis for the models
    in the model_to_run.json file.

    Args:
        start: The starting index of the model names to analyze.
        end: The ending index of the model names to analyze.

    Returns:
        response_dict: A dictionary with original (not simplified) model name:
            category set list
        costs: The cost of the analysis
    """
    with open('name_analysis/model_to_run.json', 'r', encoding='utf-8') as f:
        name_order = json.load(f)
    #name_order = sorted(name_order)
    name_order_simpl = [content.split("/")[-1] for content in name_order]
    input_ = "\n".join(name_order_simpl[start:end])
    response = chat(input_)
    ans = response.choices[0].message.content
    # calculate cost based on token
    costs = response.usage.prompt_tokens / 1000 * 0.01 + \
        response.usage.completion_tokens / 1000 * 0.03
    # split the string by \n
    separated_response = ans.split("\n")
    # create a dictionary with original (not simplified) model name: category set list
    response_dict = {}
    for i in range(len(name_order_simpl[start:end])):
        response_dict[name_order[start+i]] = get_category_list(separated_response[i])
    return response_dict, costs

def run(start: int, end: int, step: int):
    """
    This function runs the model analysis for the models
    in the model_to_run.json file.

    Args:
        start: The starting index of the model names to analyze.
        end: The ending index of the model names to analyze.
        step: The number of models to analyze in one run.

    Returns:
        None
    """
    # add variables to show time
    start_time = time.time()
    # load the results from the previous run in results.json
    try:
        with open('name_analysis/results.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except Exception: # pylint: disable=broad-except
        results = dict()
    # analyze the model using run_part
    output_dict = {}
    rerun_list = []
    i = start
    rerun_cnt = 0
    total_cost = 0
    while i < end:
        curr_start_time = time.time()
        curr_step = int(step / (2 ** rerun_cnt)) # decrease the step size if rerun_cnt > 0
        if curr_step == 0:
            curr_step = 1
        if rerun_cnt > 8:
            logger.error("Rerun count is greater than 8. Skipping")
            with open("name_analysis/rerun_index.json", "r", encoding="utf-8") as f:
                rerun_list = json.load(f)
            rerun_list.append(i)
            with open("name_analysis/rerun_index.json", "w", encoding="utf-8") as f:
                json.dump(rerun_list, f)
            i += 1
        try:
            # avoid upper bound error
            if i + curr_step > end:
                curr_step = end - i
            d, one_time_cost = run_part(i, i+curr_step)
            output_dict.update(d)
            total_cost += one_time_cost
            logger.success(f"Finished analyzing model from {i} to {i+curr_step}")
            i += curr_step
            rerun_cnt = 0
            curr_end_time = time.time()
            # update the output_dict to results
            results.update(output_dict)
            # save the results to results.json
            with open('name_analysis/results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f)
            logger.info(f"Time taken to analyze {curr_step} models:",
                f"{curr_end_time - curr_start_time} seconds")
            logger.info(f"Total cost: {total_cost}")
        except Exception as e: # pylint: disable=broad-except
            print("->", e)
            traceback_str = traceback.format_exc()
            logger.error(traceback_str)
            logger.warning(f"Error in analyzing model from {i} to {i+curr_step}")
            logger.warning(f"Step reduced, Rerunning the model from {i} to {i+curr_step}")
            rerun_cnt += 1
            curr_end_time = time.time()
            logger.info(f"Time taken to analyze {step} models:",
                f"{curr_end_time - curr_start_time} seconds")
            continue

    end_time = time.time()
    logger.info(f"Time taken to analyze {end-start} models: {end_time - start_time} seconds")


def copy_models_to_model_to_run_json(start_idx: int, end_idx: int):
    """
    This function copies the model names from the
    selected_peatmoss_repos_dn_g50.json file to the
    model_to_run.json file.

    Args:
        start_idx: The starting index of the model names to copy.
        end_idx: The ending index of the model names to copy.
    
    Returns:
        None
    """
    with open('name_analysis/model_to_run_full.json', 'r', encoding='utf-8') as f:
        name_order = json.load(f)
    name_order = name_order[start_idx:end_idx]
    with open('name_analysis/model_to_run.json', 'w', encoding='utf-8') as f:
        json.dump(name_order, f)

def model_random_shuffle():
    """
    This function shuffles the model names in the 
    selected_peatmoss_repos_dn_g50.json file and 
    saves the shuffled list to model_to_run_full.json.

    Args:
        None
    
    Returns:
        None
    """
    with open(
        'data_files/json_files/selected_peatmoss_repos_dn_g50.json', 
        'r', 
        encoding='utf-8'
    ) as f:
        name_order = json.load(f)
    random.shuffle(name_order)
    with open('name_analysis/model_to_run_full.json', 'w', encoding='utf-8') as f:
        json.dump(name_order, f)

if __name__ == '__main__':
    # model_random_shuffle()
    copy_models_to_model_to_run_json(0, 14180)
    run(0, 14180, 30)
