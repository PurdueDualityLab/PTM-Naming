# PTM-Naming

A tool to analyze defects in PTM repositories.

## Requirements

| Package        | Version |
| -------------- | ------- |
| torch          | 2.2.0   |
| torchview      | 0.2.6   |
| onnx           | 1.15.0  |
| numpy          | 1.24.4  |
| transformers   | 4.37.2  |
| pandas         | 2.0.3   |
| graphviz       | 0.20.1  |
| pillow         | 10.2.0  |
| timm           | 0.9.12  |
| scikit-learn   | 1.3.2   |
| loguru         | 0.7.2   |
| tqdm           | 4.66.1  |

## Installation

Run the following commands:
`git clone -b Evaluation https://github.com/PurdueDualityLab/PTM-Naming.git`
`pip install -r requirements.txt`
`pip install -e .`

## High level class description

### AbstractNN Class

#### Functions

#### __init__ (AbstractNN constructor)
- **Description**: The constructor of AbstractNN, not necessary to use. When this object is created, the vectorization process is automated.
- **Parameters**: 
  - `annlayer_list` (List[AbstractNNLayer]) - A list of AbstractNNLayer objects.
  - `connection_info` (List[Tuple[int, List[int]]]) - A list showing the computation graph connections.

#### from_huggingface
- **Description**: A pipeline to automatically convert huggingface model to create an AbstractNN object
- **Parameters**: 
  - `hf_repo_name` (str) - The name of the huggingface repo that contains the PTM.
  - `tracing_input` (Tensor | Tensors) - Tensor or some tensors that consists of a valid input for the specified PTM.
  - `verbose` (bool) - Controls the printing of debug messages.
- **Returns**: An AbstractNN object with the structure of specified PTM.

#### Class variables

- `layer_connection_vector` - The vector that represents the layer connections feature of the PTM.
- `layer_with_parameter_vector`  - The vector that represents the layer parameter feature of the PTM.

#### Example Usage

```python
import torch
from ANN.AbstractNN import *

ann = AbstractNN.from_huggingface(hf_repo_name, torch.randn(1, 3, 224, 224))
print(ann.layer_connection_vector)
print(ann.layer_with_parameter_vector)
```