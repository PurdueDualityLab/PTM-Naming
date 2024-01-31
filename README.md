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

```
git clone -b Evaluation https://github.com/PurdueDualityLab/PTM-Naming.git
pip install -r requirements.txt
pip install -e .
```

## High level class description

### `AbstractNN` Class

#### Class Methods

#### `AbstractNN` constructor
- **Description**: The constructor of `AbstractNN`, not necessary to use. When this object is created, the vectorization process is automated.
- **Parameters**: 
  - `annlayer_list: List[AbstractNNLayer]` - A list of `AbstractNNLayer` objects.
  - `connection_info: List[Tuple[int, List[int]]]` - A list showing the computation graph connections.

#### `from_huggingface`
- **Description**: A pipeline to automatically convert huggingface model to create an `AbstractNN` object.
- **Parameters**: 
  - `hf_repo_name: str` - The name of the huggingface repo that contains the PTM.
  - `tracing_input: str | Tensor | Tensors` - Tensor or some tensors that consists of a valid input for the specified PTM, or `auto` to automatically search for a suitable input.
  - `verbose: bool` - Controls the printing of debug messages.
- **Returns**: An `AbstractNN` object with the structure of specified PTM.

#### `from_json`
- **Description**: Export the whole ANN structure to a JSON file.
- **Parameters**: 
  - `json_loc: str` - Input JSON location.
- **Returns**: An `AbstractNN` object with the structure of specified PTM in the JSON file.

#### `export_json`
- **Description**: Export the whole ANN structure to a JSON file.
- **Parameters**: 
  - `output_loc: str` - Output JSON location.

#### Class variables

- `layer_connection_vector: dict` - The vector that represents the layer connections feature of the PTM.
- `layer_with_parameter_vector: dict`  - The vector that represents the layer parameter feature of the PTM.

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
Some weights of the model checkpoint at microsoft/resnet-18 were not used when initializing ResNetModel: ['classifier.1.bias', 'classifier.1.weight']
- This IS expected if you are initializing ResNetModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing ResNetModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2024-01-31 01:12:32.229 | SUCCESS  | ANN.AbstractNN:from_huggingface:35 - Successfully load the model.
2024-01-31 01:12:32.229 | INFO     | ANN.AbstractNN:from_huggingface:36 - Generating ANN...
Converting onnx Graph to ANN: 100%|██████████████████████████████████████████████| 3/3 [00:03<00:00,  1.30s/it]
2024-01-31 01:12:36.211 | SUCCESS  | ANN.AbstractNN:from_huggingface:53 - ANN generated. Time taken: 3.9816s
2024-01-31 01:12:36.211 | INFO     | ANN.AbstractNN:from_huggingface:54 - Vectorizing...
2024-01-31 01:12:36.213 | SUCCESS  | ANN.AbstractNN:from_huggingface:59 - Success.
{'([INPUT], Conv2d)': 1, '(Conv2d, BatchNorm2d)': 20, '(BatchNorm2d, ReLU)': 9, '(ReLU, MaxPool2d)': 1, '(MaxPool2d, Conv2d)': 1, '(MaxPool2d, add_)': 1, '(add_, ReLU)': 8, '(ReLU, Conv2d)': 18, '(ReLU, add_)': 4, '(BatchNorm2d, add_)': 11, '(ReLU, AdaptiveAvgPool2d)': 1, '(ReLU, [OUTPUT])': 1, '(AdaptiveAvgPool2d, [OUTPUT])': 1}
{'[INPUT]': 1, "Conv2d ['<in_channels, 3>', '<out_channels, 64>', '<kernel_size, (7, 7)>', '<stride, (2, 2)>', '<padding, (3, 3)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 64>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "ReLU ['<inplace, False>']": 17, "MaxPool2d ['<kernel_size, 3>', '<stride, 2>', '<padding, 1>', '<dilation, 1>', '<return_indices, False>', '<ceil_mode, False>']": 1, 'add_ ': 8, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 128>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 256>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 256>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 512>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 512>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "AdaptiveAvgPool2d ['<output_size, (1, 1)>']": 1, '[OUTPUT]': 2, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 128>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 64>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 4}
```

#### Example Usage (Automatic Input Prepare)

```python
import torch
from ANN.AbstractNN import *

ann = AbstractNN.from_huggingface(hf_repo_name, torch.randn(1, 3, 224, 224))
print(ann.layer_connection_vector)
print(ann.layer_with_parameter_vector)
```

#### Example Result

```
2024-01-31 04:11:38.408 | INFO     | ANN.AbstractNN:from_huggingface:27 - Looking for model in microsoft/resnet-18...
Some weights of the model checkpoint at microsoft/resnet-18 were not used when initializing ResNetModel: ['classifier.1.bias', 'classifier.1.weight']
- This IS expected if you are initializing ResNetModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing ResNetModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
2024-01-31 04:11:45.006 | SUCCESS  | ANN.AbstractNN:from_huggingface:30 - Successfully load the model.
2024-01-31 04:11:45.007 | INFO     | ANN.AbstractNN:from_huggingface:33 - Automatically generating an input...
/home/cheung59/.local/lib/python3.8/site-packages/transformers/models/convnext/feature_extraction_convnext.py:28: FutureWarning: The class ConvNextFeatureExtractor is deprecated and will be removed in version 5 of Transformers. Please use ConvNextImageProcessor instead.
  warnings.warn(
Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.
Could not find image processor class in the image processor config or the model config. Loading based on pattern matching with the model's feature extractor configuration.
  0%|                                                                                                                   | 0/9 [00:00<?, ?it/s]2024-01-31 04:12:17.105 | SUCCESS  | tools.HFValidInputIterator:get_valid_input:46 - Find an input for microsoft/resnet-18
 11%|███████████▉                                                                                               | 1/9 [00:31<04:08, 31.04s/it]
2024-01-31 04:12:17.204 | SUCCESS  | ANN.AbstractNN:from_huggingface:36 - Successfully generating an input.
2024-01-31 04:12:17.205 | INFO     | ANN.AbstractNN:from_huggingface:39 - Generating ANN...
Converting onnx Graph to ANN: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.42s/it]
2024-01-31 04:12:21.464 | SUCCESS  | ANN.AbstractNN:from_huggingface:56 - ANN generated. Time taken: 4.2592s
2024-01-31 04:12:21.464 | INFO     | ANN.AbstractNN:from_huggingface:57 - Vectorizing...
2024-01-31 04:12:21.466 | SUCCESS  | ANN.AbstractNN:from_huggingface:62 - Success.
{'([INPUT], Conv2d)': 1, '(Conv2d, BatchNorm2d)': 20, '(BatchNorm2d, ReLU)': 9, '(ReLU, MaxPool2d)': 1, '(MaxPool2d, Conv2d)': 1, '(MaxPool2d, add_)': 1, '(ReLU, Conv2d)': 18, '(BatchNorm2d, add_)': 11, '(add_, ReLU)': 8, '(ReLU, add_)': 4, '(ReLU, AdaptiveAvgPool2d)': 1, '(ReLU, [OUTPUT])': 1, '(AdaptiveAvgPool2d, [OUTPUT])': 1}
{'[INPUT]': 1, "Conv2d ['<in_channels, 3>', '<out_channels, 64>', '<kernel_size, (7, 7)>', '<stride, (2, 2)>', '<padding, (3, 3)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 64>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "ReLU ['<inplace, False>']": 17, "MaxPool2d ['<kernel_size, 3>', '<stride, 2>', '<padding, 1>', '<dilation, 1>', '<return_indices, False>', '<ceil_mode, False>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 64>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 4, 'add_ ': 8, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 128>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 128>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 256>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 512>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, '[OUTPUT]': 2, "AdaptiveAvgPool2d ['<output_size, (1, 1)>']": 1, "Conv2d ['<in_channels, 512>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 256>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1}
```

The two class variables can be directly converted to JSON using:

```python
import json

with open("path/to/json1", "w") as f:
    json.dump(ann.layer_connection_vector, f)
with open("path/to/json2", "w") as f:
    json.dump(ann.layer_with_parameter_vector, f)
```

### `HFValidInputIterator` Class

#### Class Methods

#### `HFValidInputIterator` constructor
- **Description**: The constructor of `AbstractNN`, not necessary to use. When this object is created, the vectorization process is automated.
- **Parameters**: 
  - `model: Any` - A Hugging Face model obtained from `AutoModel.from_pretrained()`.
  - `hf_repo_name: str` - Name of the target Hugging Face repository.
  - `cache_dir: str` - Directory to cache the Hugging Face repository clones, can be `None`.

#### `get_valid_input`
- **Description**: A pipeline to automatically return a valid input for the given `hf_repo_name`.
- **Returns**: An valid input object that can be used for tracing the computational graph of the PTM.

#### Example Usage

```python
from transformers import AutoModel
from tools.HFAutoClassIterator import HFAutoClassIterator

repo_name = "anton-l/distilhubert-ft-keyword-spotting"
model = AutoModel.from_pretrained(repo_name)
in_iter = HFValidInputIterator(model, repo_name, None)
print(in_iter.get_valid_input())
```

#### Example Result

```
 43%|█████████████████████████████████████████████▊                                                             | 3/7 [00:23<00:31,  7.96s/it]
{'input_values': tensor([[-0.4477, -0.9134,  0.6262,  ..., -0.8534,  1.3059,  0.4376]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], dtype=torch.int32)}
```
