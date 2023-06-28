import os
import sys
import json
import transformers

from loguru import logger

logger.remove()  # Remove all handlers
logger.add(sys.stderr, level='INFO')  # Add a new handler for INFO and above


def load_metadata(path='./model_metadata/model_archs.json') -> dict:
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        metadata_list = json.load(f)
    return metadata_list
    
def load_model_list(path='./modelArch_list.json'):
    '''
    Load model list from model_list.json
    '''
    with open(os.path.join(os.path.dirname(__file__), path), "r") as f:
        model_list = json.load(f)
    return model_list


def get_model_from_arch(model_arch: str, metadata_dict: dict):
    '''
    Get all model from huggingface with the same model architecture
    The models are obtained from model metadata.
    ''' 
    if model_arch not in metadata_dict:
        logger.debug(f'No model for {model_arch}, continue to next model.')
    else:
        model_metadata = metadata_dict[model_arch]
        logger.info(f'Loading {len(model_metadata)} {model_arch} models.')
        # Saving the model_archs with more than 2 models
        if len(model_metadata) > 1:
            return model_metadata
        else:
            logger.info(f'Only 1 model for {model_arch}, continue to next model.')
    return None


def load_model(model_list: list, metadata_dict: dict) -> None:
    '''
    Load model with same architecture from huggingface
    '''
    filtered_models = {}
    for model_arch in model_list:
        models_metadata = get_model_from_arch(model_arch, metadata_dict)
    
        if models_metadata:
            logger.info(f'Saving {model_arch} to filtered_models')
            filtered_models[model_arch] = models_metadata
        
    logger.info(f'Number of filtered_models: {len(filtered_models)}')
    logger.success(f'Saving filtered_models to filtered_models.json')
    with open(os.path.join(os.path.dirname(__file__), 'filtered_models.json'), 'w') as f:
        json.dump(filtered_models, f, indent=4)
    return


if __name__=='__main__':
    model_list = load_model_list()
    metadata_dict = load_metadata()
    logger.info(f'Number of models: {len(model_list)}')
    logger.info(f'Number of metadata: {len(metadata_dict)}')

    load_model(model_list, metadata_dict)


