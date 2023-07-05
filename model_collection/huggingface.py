import os
import sys
import json
from tqdm import tqdm

import transformers
from huggingface_hub import hf_hub_download, HfApi, ModelFilter, list_models

from loguru import logger

logger.remove()  # Remove all handlers
logger.add(sys.stderr, level='INFO')  # Add a new handler for INFO and above

api = HfApi()
models = api.list_models()

def load_metadata(path='./model_metadata/model_archs.json') -> dict:
    '''
    Load model metadata from model_archs.json
    '''
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        metadata_list = json.load(f)
    return metadata_list
    

def load_model_list(path='./modelArch_list.json') -> list:
    '''
    Load model list from model_list.json
    '''
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        model_list = json.load(f)
    return model_list


def get_downloads(model_name):    
    # Use hfapi to get the model from model_name)
    try:
        return api.model_info(repo_id=model_name).downloads
    except:
        return None
        

def get_model_from_arch(model_arch: str, metadata_dict: dict):
    '''
    Get all model from huggingface with the same model architecture
    The models are obtained from model metadata.
    ''' 
    

    if model_arch not in metadata_dict:
        logger.debug(f'No model for {model_arch}, continue to next model.')
    else:
        model_metadata = metadata_dict[model_arch]
        model_metadata_nameDownload = model_metadata.copy()
        
        # Get the model downloads
        for model in model_metadata:
            model_download = get_downloads(model)
            if model_download is None:
                logger.debug(f'{model} does not exist, continue to next model.')
                model_metadata.remove(model)
                model_metadata_nameDownload.remove(model)
            elif model_download < 10:
                logger.debug(f'{model} has less than 10 downloads, continue to next model.')
                model_metadata.remove(model)
                model_metadata_nameDownload.remove(model)
            else:
                # replace mdoel with (model, model_download) in model_metadata
                model_metadata_nameDownload[model_metadata.index(model)] = (model, model_download)
        logger.debug(f'Loading {len(model_metadata)} {model_arch} models.')
        # Saving the model_archs with more than 2 models
        if len(model_metadata) > 1:
            return model_metadata, model_metadata_nameDownload
        else:
            logger.debug(f'Only 1 model for {model_arch}, continue to next model.')
    return None, None


def filter_downloads(filtered_models: dict) -> dict:
    '''
        TODO: Remove. Use hf_download instead
    '''
    # Load filtered_models
    with open(os.path.join(os.path.dirname(__file__), 'filtered_models.json'), "r") as f:
        filtered_models = json.load(f)
    
    for model_arch in filtered_models:
        for model in filtered_models[model_arch]:
            logger.error(model)
            sys.exit()
    return filtered_models


def load_model(model_list: list, metadata_dict: dict):
    '''
    Load model with same architecture from huggingface
    '''
    filtered_models = {}
    filtered_models_nameDownload = {}

    for model_arch in tqdm(model_list):
         
        models_metadata, model_metadata_nameDownload = get_model_from_arch(model_arch, metadata_dict)
       
    
        if models_metadata:
            logger.debug(f'Saving {model_arch} to filtered_models_nameOnly')
            filtered_models[model_arch] = models_metadata
        
        if model_metadata_nameDownload:
            logger.debug(f'Saving {model_arch} to filtered_models_nameDownload')
            filtered_models_nameDownload[model_arch] = model_metadata_nameDownload
        
    logger.info(f'Number of filtered_models: {len(filtered_models)}')
    logger.info(f'Number of filtered_models_nameDownload: {len(filtered_models_nameDownload)}')
    # filtered_models = filter_downloads(filtered_models)
    
    with open(os.path.join(os.path.dirname(__file__), 'filtered_models.json'), 'w') as f:
        json.dump(filtered_models, f, indent=4)
    logger.success(f'Saved filtered_models!')

    with open(os.path.join(os.path.dirname(__file__), 'filtered_models_nameDownload.json'), 'w') as f:
        json.dump(filtered_models_nameDownload, f, indent=4)
    logger.success(f'Saved filtered_models_nameDownload!')



    return


def download_model(filtered_models='filtered_models.json', download_path=''):
    # load filtered_models
    with open(os.path.join(os.path.dirname(__file__), filtered_models), "r") as f:
        filtered_models = json.load(f)
    
    for model_arch in filtered_models:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), download_path, model_arch)):
            os.makedirs(os.path.join(os.path.dirname(__file__), download_path, model_arch))
    
        for model in filtered_models[model_arch]:
            logger.info(f'Downloading {model}')
            hf_hub_download(repo_id=model, filename="config.json")
    return

if __name__=='__main__':
    model_list = load_model_list()
    metadata_dict = load_metadata()
    logger.info(f'Number of models: {len(model_list)}')
    logger.info(f'Number of metadata: {len(metadata_dict)}')

    load_model(model_list, metadata_dict)

