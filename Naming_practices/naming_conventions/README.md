## Overview
This section aims to improve the accuracy of model naming categorization using OpenAI's GPT-4 API by experimenting with different prompt structures and data formats.

## Data Organization
The `data` folder contains two sub-folders with input data in different formats:
- `list_per_message`: Contains data where each message is sent as a list.
- `one_per_message`: Contains data where each message is sent individually.

## Prompts
The `prompt` folder includes files with prefixes `const` followed by a number. Each file represents an example prompt that is refined for specific feature used in the experiments.

## Experiments
`openai_response.ipynb` is the main notebook containing the code to interact with OpenAI's API. It includes:
- Initialization of the OpenAI client with API key and organization.
- Use of the `.chat.completions.create` method with different temperatures to generate predictions.
- Code for computing and displaying the confusion matrix and classification report.

## Usage
To run the main notebook, you need to:
1. Install the necessary libraries listed in the notebook.
2. Set up an `.env` file with your OpenAI API key and organization.
3. Execute the cells in `openai_response.ipynb` to generate predictions and evaluate the model.

## Results
The experiments' results, such as accuracy improvements, are visually represented using plots generated from the data provided.

## Steps that we take to get to the result
Will be updated Soon