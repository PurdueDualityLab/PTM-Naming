import os
import json
import transformers

from loguru import logger


# Load model list from json file
def load_model_list():
    with open(os.path.join(os.path.dirname(__file__), "model_list.json"), "r") as f:
        model_list = json.load(f)
    return model_list

if __name__=='__main__':
    model_list = load_model_list()
    logger.info(f'Number of models: {len(model_list)}')
