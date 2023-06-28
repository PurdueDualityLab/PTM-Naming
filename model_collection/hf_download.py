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
            if not os.path.exists(os.path.join(os.path.dirname(__file__), download_path, model_arch, model)):
                os.makedirs(os.path.join(os.path.dirname(__file__), download_path, model_arch, model))
            try:
                # download the config file from huggingface
                config_file = hf_hub_download(repo_id=model, filename="config.json")
                # save the config_file
                with open(os.path.join(os.path.dirname(__file__), download_path, model_arch, model, 'config.json'), 'w') as f:
                    json.dump(config_file, f, indent=4)
                if not metadata_only:
                    torch_model = hf_hub_download(repo_id=model, filename="pytorch_model.bin")
                    # save the torch_model
                    with open(os.path.join(os.path.dirname(__file__), download_path, model_arch, model, 'pytorch_model.bin'), 'wb') as f:
                        f.write(torch_model)
            except:
                logger.error(f'Error downloading {model}.\n')
                continue
    return


if __name__=='__main__':
    download_model(metadata_only=True)