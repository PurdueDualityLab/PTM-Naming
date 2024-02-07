import os
import prompt
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM

from loguru import logger

os.environ["TRANSFORMERS_CACHE"] = "/scratch/gilbreth/jiang784/.cache/huggingface"

# Load codellama/CodeLlama-13b-Instruct-hf
logger.info("Loading model codellama/CodeLlama-13b-Instruct-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")


logger.info("Loading tokenizer codellama/CodeLlama-13b-Instruct-hf")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-13b-Instruct-hf")

# Print the model architecture
logger.info("Loaded model: codellama/CodeLlama-13b-Instruct-hf")

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    # Wrap the model with nn.DataParallel
    model = torch.nn.DataParallel(model)

# Move the model to GPU(s)
model.cuda() 

prompt = prompt.BACKGROUND
inputs = tokenizer(f"{prompt} \n Input: CodeLlama-13b-Instruct-hf", return_tensors="pt").to('cuda')

logger.info("Generating text based on the provided prompt")

# Assuming you have large inputs and want to generate a reasonable amount of text
max_input_length = 512  # Maximum input length the model can handle
max_new_tokens = 256  # Maximum new tokens to generate

# Adjusting inputs if necessary
if len(inputs['input_ids'][0]) > max_input_length:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to('cuda')

# Generating text with adjusted parameters
output = model.module.generate(**inputs, 
                               max_length=max_input_length + max_new_tokens,  # Adjusting total length
                               max_new_tokens=max_new_tokens,  # New parameter to control generation length
                               num_return_sequences=1)
logger.info("Decoding the generated text")
logger.debug(f"Generated text: {output}")
# Decode the generated text
decoded_output = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)

print(decoded_output)