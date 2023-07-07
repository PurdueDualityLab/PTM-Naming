import os
import sys
import json
from tqdm import tqdm

import transformers
from huggingface_hub import hf_hub_download

from loguru import logger

logger.remove()  # Remove all handlers
logger.add(sys.stderr, level='INFO')  # Add a new handler for INFO and above


def download_model(filtered_models='filtered_models.json', download_path='./downloads', metadata_only=True):
    # load filtered_models
    with open(os.path.join(os.path.dirname(__file__), filtered_models), "r") as f:
        filtered_models = json.load(f)
    
    tqdm.write(f'Downloading {len(filtered_models)} models.')
    for model_arch in tqdm(filtered_models):
        if not os.path.exists(os.path.join(os.path.dirname(__file__), download_path, model_arch)):
            os.makedirs(os.path.join(os.path.dirname(__file__), download_path, model_arch))

        tqdm.write(f'Downloading {len(filtered_models[model_arch])} {model_arch} models.')
        for model in tqdm(filtered_models[model_arch]):
            # logger.info(f'Downloading {model}')
            # if not os.path.exists(os.path.join(os.path.dirname(__file__), download_path, model_arch, model)):
            model_path = os.path.join(os.path.dirname(__file__), download_path, model_arch, model)
                # os.makedirs(model_path)
            try:
                # Download the config file from huggingface
                hf_hub_download(local_dir=model_path, repo_id=model, filename="config.json")
            except:
                logger.error(f'Error downloading {model} config file.\n')
            
            if not metadata_only:
                # Download the pytorch model from huggingface
                try:
                    hf_hub_download(local_dir=model_path, repo_id=model, filename="pytorch_model.bin")
                except:
                    logger.error(f'Error downloading {model} pytorch model.\n')
        break
    return


if __name__=='__main__':
    download_model(metadata_only=True)