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

#### AbstractNN constructor
- **Description**: The constructor of AbstractNN, not necessary to use. When this object is created, the vectorization process is automated.
- **Parameters**: 
  - `annlayer_list: List[AbstractNNLayer]` - A list of AbstractNNLayer objects.
  - `connection_info: List[Tuple[int, List[int]]]` - A list showing the computation graph connections.

#### from_huggingface
- **Description**: A pipeline to automatically convert huggingface model to create an AbstractNN object
- **Parameters**: 
  - `hf_repo_name: str` - The name of the huggingface repo that contains the PTM.
  - `tracing_input: Tensor | Tensors` - Tensor or some tensors that consists of a valid input for the specified PTM.
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

#### Example Result

```
2024-01-31 01:11:50.609 | INFO     | ANN.AbstractNN:from_huggingface:32 - Looking for model in microsoft/resnet-18...
2024-01-31 01:12:32.229 | SUCCESS  | ANN.AbstractNN:from_huggingface:35 - Successfully load the model.
2024-01-31 01:12:32.229 | INFO     | ANN.AbstractNN:from_huggingface:36 - Generating ANN...
Converting onnx Graph to ANN: 100%|██████████████████████████████████████████████| 3/3 [00:03<00:00,  1.30s/it]
2024-01-31 01:12:36.211 | SUCCESS  | ANN.AbstractNN:from_huggingface:53 - ANN generated. Time taken: 3.9816s
2024-01-31 01:12:36.211 | INFO     | ANN.AbstractNN:from_huggingface:54 - Vectorizing...
2024-01-31 01:12:36.213 | SUCCESS  | ANN.AbstractNN:from_huggingface:59 - Success.
{'([INPUT], Conv2d)': 1, '(Conv2d, BatchNorm2d)': 20, '(BatchNorm2d, ReLU)': 9, '(ReLU, MaxPool2d)': 1, '(MaxPool2d, Conv2d)': 1, '(MaxPool2d, add_)': 1, '(add_, ReLU)': 8, '(ReLU, Conv2d)': 18, '(ReLU, add_)': 4, '(BatchNorm2d, add_)': 11, '(ReLU, AdaptiveAvgPool2d)': 1, '(ReLU, [OUTPUT])': 1, '(AdaptiveAvgPool2d, [OUTPUT])': 1}
{'[INPUT]': 1, "Conv2d ['<in_channels, 3>', '<out_channels, 64>', '<kernel_size, (7, 7)>', '<stride, (2, 2)>', '<padding, (3, 3)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 64>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "ReLU ['<inplace, False>']": 17, ...}
```