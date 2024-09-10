import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

import json
import constant
import time
import os

from loguru import logger

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID")
)
logger.remove(0)
logger.add("prediction.log")

def open_ai(pred, naming_category, file_path = "data/model_to_run_full.jsonl", temp = 0.1):
    thres = 90
    gpt_error = False

    with open(file_path, "r") as file:
        lines = file.readlines()
    start = time.time()
    logger.success(f'Started at time {start}')
    for i, line in enumerate(lines):
        if i < 4:   #skip lines 0-3 (2000 dataset)
            continue
        obj = json.loads(line)
        messages = obj["messages"]
        user_contents = messages[0]["content"]
        
        logger.success(f'Beginning of user_content: {user_contents[0]}')
        for user_content in user_contents:
            try:
                chatlog = []
                chatlog.append({"role": "system", "content": constant.BACKGROUND})
                chatlog.append({"role": "user", "content": user_content})
                completion = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=chatlog,
                    temperature=temp,
                    logprobs=True,
                    top_logprobs=2
                )
                output = completion.choices[0].message.content
                prob = np.round(np.exp(completion.choices[0].logprobs.content[0].logprob)*100,2)
                if prob > thres and output in naming_category:
                    pred[output] += 1
                    
            except Exception as e:
                logger.error(f'GPT4 unable to generate response due to {e}')
                gpt_error = True
                break
        if gpt_error:
            logger.error(f'*** Breaking out of loop due to error ***')
            logger.error(f'Terminated from line {i}')
            logger.error(f'Beginning of user_content: {user_contents[0]}')
            logger.info(f'Prediction so far: {pred}')
            break
        logger.success(f'Time elapsed: {time.time()-start}. Prediction so far: {pred} @line {i}')
        if i == 5: # testing first 2000 models
            break
    return pred
def run():
    start = time.time()
    
    naming_category = ['Application and Task', 'Implementation', 'Implementation with Application and Task', 'Other']
    pred = {'Application and Task': 79, 'Implementation': 1076, 'Implementation with Application and Task': 504, 'Other': 204}  # load pred if terminated due to Error
    
    pred = open_ai(pred, naming_category)
    
    logger.success(f'Prediction running all jsonl entry: {pred}')
    logger.success(f'Total time elapsed: {time.time()-start}')
    
    return pred

def pred_plot(pred):
    naming_category = ['Impl. +\n App./Task', 'Impl.\nunit', 'App./Task', 'Other']
    
    survey_data = [106, 26, 41, 3]
    survey_total = sum(survey_data)
    survey_percentage = [np.round(s/survey_total*100) for s in survey_data]
    
    total = sum(pred.values())
    percentage = [np.round(p/total*100) for p in pred.values()]
    pred = [p for p in pred.values()]
    
    N = 4
    ind = np.arange(0,N)
    width = 0.4
    
    plt.figure()
    plt.box(False)
    plt.xticks(ind, naming_category, fontsize=18)
    plt.bar(ind-width/2, survey_percentage, width=width, zorder=2, label="Survey data")
    plt.bar(ind+width/2, percentage, width=width, zorder=2, label="Practical data")
    plt.ylabel("Frequency (%)", fontsize=19)
    plt.grid(which='major', axis='y', zorder=1)
    plt.tight_layout()
    max_value = max(max(survey_percentage, percentage))
    plt.ylim(top=max_value*1.1)
    plt.legend(loc='best', fontsize=18)
    plt.savefig('plot/Survey_Practical_figure.pdf')
    
if __name__ == "__main__":
    # pred = run()
    pred = {'Implementation with Application and Task': 736, 'Implementation': 1628, 'Application and Task': 118, 'Other': 320}
    pred_plot(pred)