import os
import prompt
import torch
import json
import transformers
from transformers import AutoTokenizer, LlamaModel

from loguru import logger

os.environ["TRANSFORMERS_CACHE"] = "/scratch/gilbreth/jiang784/.cache/huggingface"

def get_name_embeddings(models, save_results=False):
    # Assuming 'llama/checkpoint-13b' as the model checkpoint
    model_name = "codellama/CodeLlama-13b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlamaModel.from_pretrained(model_name)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        model.cuda()
        logger.info("Model moved to GPU for faster processing.")
    else:
        logger.warning("CUDA is not available. Running on CPU...")

    model.eval()  # Set the model to evaluation mode
    
    embeddings = {}
    for arch in models:
        for model_name in models[arch]:
            if len(model_name.split("/")) > 1:
                model_name = model_name.split("/")[-1]
            logger.info(f"Creating embedding for model {model_name}")
            
            # Tokenize the model name
            inputs = tokenizer(model_name, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
            
            # Obtain embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use mean pooling over the last hidden states for the embedding
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            embeddings[model_name] = embedding.flatten()  # Flatten the embedding
            
        break  # This break is to exit after the first architecture, as in the original code

    if save_results:
        # Save embeddings to file
        with open("embeddings.json", "w") as f:
            json.dump(embeddings, f, cls=NumpyEncoder)  # Ensure numpy arrays are saved properly

    return embeddings

