import json

from loguru import logger

def model_count(model_list_path='./filtered_models.json', end_with_model=False):
    """
    Count the number of unique models in the model list.
    """

    # Load model list from json file
    with open(model_list_path, 'r') as f:
        filtered_models = json.load(f)

    # Initialize counters and list for model counts
    total_model_count = 0
    model_type_count = 0
    model_counts = []

    # Iterate over each model type
    for model_type in filtered_models:
        # Check if model type ends with 'Model' if required
        if not end_with_model or model_type.endswith('Model'):
            # Get unique models of this type
            unique_models = set(model for model in filtered_models[model_type] if model is not None)

            # Update counters
            model_count = len(unique_models)
            if model_count > 0:
                model_type_count += 1
                total_model_count += model_count
                model_counts.append(model_count)
                logger.info(f'{model_type}: {model_count}')
    
    # Calculate and log range of model counts
    range_counts = max(model_counts) - min(model_counts) if model_counts else 0
    logger.info(f'end_with_model is {end_with_model}')
    logger.info(f'Max #models: {max(model_counts)}, Min #models: {min(model_counts)}, Range: {range_counts}')
    logger.info(f'Total #models: {total_model_count}, #Model types: {model_type_count}\n')


if __name__ == "__main__":
    model_count(end_with_model=False)
    model_count(end_with_model=True)
