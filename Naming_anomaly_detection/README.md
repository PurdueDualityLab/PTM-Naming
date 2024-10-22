## Overview

This folder includes all the scripts and data used in our DNN Architecture Assessment and Contrastive Learning pipeline (DARA, CL, RQ4).

Larger version of figures are available: 
- [Figure 8](/Naming_anomaly_detection/DARA/results/PCA_projection_model_type.pdf): visualization of selected PTM features based on `model_type`.
- [Figure 9](/Naming_anomaly_detection/DARA/results/confusion_matrix_heatmap_model_type.pdf): Confusion matrix of evaluation results on `model_type`.
- [Figure 10](/Naming_anomaly_detection/DARA/results/confusion_matrix_heatmap_task.pdf): Confusion matrix of evaluation results on `task`.
- [Figure 11](/Naming_anomaly_detection/DARA/results/confusion_matrix_heatmap_arch.pdf):  Confusion matrix of evaluation results on `arch`.

## Requirements

| Package        | Version |
| -------------- | ------- |
| torch          | 2.2.0   |
| torchview      | 0.2.6   |
| onnx           | 1.15.0  |
| numpy          | 1.24.4  |
| transformers   | 4.41.2  |
| pandas         | 2.0.3   |
| graphviz       | 0.20.1  |
| pillow         | 10.2.0  |
| timm           | 0.9.12  |
| scikit-learn   | 1.3.2   |
| loguru         | 0.7.2   |
| tqdm           | 4.66.1  |
| python-dotenv  | 1.0.1   |
| matplotlib     | 3.7.4   |
| openai         | 1.12.0  |
| bitsandbytes    | 0.42.0  |
| accelerate      | 0.27.2  |
| openpyxl        | 3.1.2   |
| fugashi         | 1.3.2   |
| unidic-lite     | 1.0.8   |
| ipadic          | 1.0.0   |
| auto-gptq       | latest  |


## Installation

Run the following commands:

```
git clone -b https://github.com/PurdueDualityLab/PTM-Naming.git
pip install -r requirement.txt
export PYTHONPATH="${PYTHONPATH}:absolute/path/to/PTM-Naming"
```

<!-- ## TODOs

1. Change the dummy input to non-random inputs for `tools/HFAutoClassIterator.py` and `tools/HFValidInputIterator.py`.
2. Test `APTM/abstract_neural_network.py:265`. Evaluate results in other n-grams. -->

## High level class description

### `APTM.abstract_neural_network.AbstractNN` Class

#### Class Methods

#### `AbstractNN` constructor
- **Description**: The constructor of `AbstractNN`, not necessary to use. When this object is created, the vectorization process is automated.
- **Parameters**: 
  - `aptmlayer_list: List[AbstractNNLayer]` - A list of `AbstractNNLayer` objects.
  - `connection_info: List[Tuple[int, List[int]]]` - A list showing the computation graph connections.

#### `from_huggingface`
- **Description**: A pipeline to automatically convert huggingface model to create an `AbstractNN` object.
- **Parameters**: 
  - `hf_repo_name: str` - The name of the huggingface repo that contains the PTM.
  - `tracing_input: str | Tensor | Tensors` - Tensor or some tensors that consists of a valid input for the specified PTM, `auto` to automatically search for a suitable input, default value is `auto`.
  - `verbose: bool` - Controls the printing of debug messages.
- **Returns**: An `AbstractNN` object with the structure of specified PTM.

#### `from_json`
- **Description**: Load the whole APTM structure from a JSON file.
- **Parameters**: 
  - `json_loc: str` - Input JSON location.
- **Returns**: An `AbstractNN` object with the structure of specified PTM in the JSON file.

#### `export_json`
- **Description**: Export the whole APTM structure to a JSON file.
- **Parameters**: 
  - `output_loc: str` - Output JSON location.

#### Class variables

- `layer_connection_vector: dict` - The vector that represents the layer connections feature of the PTM.
- `layer_with_parameter_vector: dict`  - The vector that represents the layer parameter feature of the PTM.

#### Example Usage

```python
import torch
from abstract_neural_network.AbstractNN import *

aptm = AbstractNN.from_huggingface(hf_repo_name, torch.randn(1, 3, 224, 224))
print(aptm.layer_connection_vector)
print(aptm.layer_with_parameter_vector)
```

#### Example Result

```
{'([INPUT], Conv2d)': 1, '(Conv2d, BatchNorm2d)': 20, '(BatchNorm2d, ReLU)': 9, '(ReLU, MaxPool2d)': 1, '(MaxPool2d, Conv2d)': 1, '(MaxPool2d, add_)': 1, '(add_, ReLU)': 8, '(ReLU, Conv2d)': 18, '(ReLU, add_)': 4, '(BatchNorm2d, add_)': 11, '(ReLU, AdaptiveAvgPool2d)': 1, '(ReLU, [OUTPUT])': 1, '(AdaptiveAvgPool2d, [OUTPUT])': 1}
{'[INPUT]': 1, "Conv2d ['<in_channels, 3>', '<out_channels, 64>', '<kernel_size, (7, 7)>', '<stride, (2, 2)>', '<padding, (3, 3)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 64>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "ReLU ['<inplace, False>']": 17, "MaxPool2d ['<kernel_size, 3>', '<stride, 2>', '<padding, 1>', '<dilation, 1>', '<return_indices, False>', '<ceil_mode, False>']": 1, 'add_ ': 8, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 128>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 256>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 256>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 512>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 512>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "AdaptiveAvgPool2d ['<output_size, (1, 1)>']": 1, '[OUTPUT]': 2, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 128>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 64>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 4}
```

#### Example Usage (Automatic Input Prepare)

```python
import torch
from abstract_neural_network.AbstractNN import *

aptm = AbstractNN.from_huggingface(hf_repo_name)
print(aptm.layer_connection_vector)
print(aptm.layer_with_parameter_vector)
```

#### Example Result

```
{'([INPUT], Conv2d)': 1, '(Conv2d, BatchNorm2d)': 20, '(BatchNorm2d, ReLU)': 9, '(ReLU, MaxPool2d)': 1, '(MaxPool2d, Conv2d)': 1, '(MaxPool2d, add_)': 1, '(ReLU, Conv2d)': 18, '(BatchNorm2d, add_)': 11, '(add_, ReLU)': 8, '(ReLU, add_)': 4, '(ReLU, AdaptiveAvgPool2d)': 1, '(ReLU, [OUTPUT])': 1, '(AdaptiveAvgPool2d, [OUTPUT])': 1}
{'[INPUT]': 1, "Conv2d ['<in_channels, 3>', '<out_channels, 64>', '<kernel_size, (7, 7)>', '<stride, (2, 2)>', '<padding, (3, 3)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 64>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "ReLU ['<inplace, False>']": 17, "MaxPool2d ['<kernel_size, 3>', '<stride, 2>', '<padding, 1>', '<dilation, 1>', '<return_indices, False>', '<ceil_mode, False>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 64>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 4, 'add_ ': 8, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 128>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 128>', '<out_channels, 128>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 256>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "BatchNorm2d ['<num_features, 512>', '<eps, 1e-05>', '<momentum, 0.1>', '<affine, True>', '<track_running_stats, True>']": 5, '[OUTPUT]': 2, "AdaptiveAvgPool2d ['<output_size, (1, 1)>']": 1, "Conv2d ['<in_channels, 512>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 256>', '<out_channels, 512>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 256>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (1, 1)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 3, "Conv2d ['<in_channels, 128>', '<out_channels, 256>', '<kernel_size, (3, 3)>', '<stride, (2, 2)>', '<padding, (1, 1)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1, "Conv2d ['<in_channels, 64>', '<out_channels, 128>', '<kernel_size, (1, 1)>', '<stride, (2, 2)>', '<padding, (0, 0)>', '<dilation, (1, 1)>', '<transposed, False>', '<output_padding, (0, 0)>', '<groups, 1>', '<padding_mode, zeros>']": 1}
```

The two class variables can be directly converted to JSON using:

```python
import json

with open("path/to/json1", "w") as f:
    json.dump(aptm.layer_connection_vector, f)
with open("path/to/json2", "w") as f:
    json.dump(aptm.layer_with_parameter_vector, f)
```

### `tools.HFValidInputIterator` Class

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

### `vector.ClusterPipeline` Class

#### Class Methods

#### `ClusterPipeline` constructor (No parameter)
- **Description**: The constructor of `ClusterPipeline`, automatically creates `self.cluster_data` which points toward the internal clustering directory specified by the `.env` file `CLUSTER_DIR` variable. The `CLUSTER_DIR` is a directory that contains 6 files: `k_l.pkl`, `k_p.pkl`, `k_d.pkl`, `vec_l.pkl`, `vec_p.pkl`, `vec_d.pkl`.

#### `cluster_with_extra_model`
- **Description**: A pipeline to automatically cluster a single model with a group of models in the internal cluster.
- **Parameters**: 
  - `arch_name: str` - The name of the specified architecture, must be one of the architectures in the internal cluster.
  - `additional_model_vector: APTMVectorTriplet` - An `APTMVectorTriplet` class object to be an extra model to cluster with the ones that already exist in the internal cluster.
  - `eps: int` - Controls the strictness of the clustering.
- **Returns**: A tuple with `results` and `outliers`.

#### `cluster_with_extra_model_from_huggingface`
- **Description**: A pipeline to automatically cluster a single hugging face model with a group of models in the internal cluster.
- **Parameters**: 
  - `hf_repo_name: str` - The name of the hugging face repository.
  - `arch_name: str` - The name of the architecture, defaults to `auto` so it automatically gets the architecture from hugging face `config.json`.
  - `model_name: str` - The custom name of the model, defaults to `auto` so it automatically gets the model name from hugging face repository.
  - `eps: int` - Controls the strictness of the clustering.
- **Returns**: A tuple with `results` and `outliers`.

### `vector.aptm_vector.APTMVectorTriplet` Class

#### Class Methods

#### `from_APTM` (static method)
- **Description**: A function that converts an `AbstractNN` object to an `APTMVectorTriplet` object.
- **Parameters**: 
  - `model_name: str` - The name of the model.
  - `aptm: AbstractNN` - The `AbstractNN` object to be converted.
- **Returns**: An `APTMVectorTriplet` object.

#### Example Usage

```python
from vector.APTMVector import APTMVectorTriplet
from vector.ClusterPipeline import ClusterPipeline
# An AbstractNN object 'my_aptm' is already defined

my_aptm_vector_triplet = APTMVectorTriplet.from_APTM(my_aptm)
res, out = ClusterPipeline().cluster_with_extra_model("DesiredArchitecture", my_aptm_vector_triplet)
```

#### Or Simply 

```python
print(ClusterPipeline().cluster_with_extra_model_from_huggingface("microsoft/resnet-18"))
```

#### Example Results

```
({'ResNet': {'0': ['microsoft/resnet-50', 'keithanpai/resnet-50-finetuned-eurosat', 'jayanta/microsoft-resnet-50-cartoon-face-recognition', 'arize-ai/resnet-50-fashion-mnist-quality-drift', 'BirdL/CatsandDogsPOC-Resnet', 'AlexKolosov/my_first_model', 'jayanta/microsoft-resnet-50-cartoon-emotion-detection', 'arize-ai/resnet-50-cifar10-quality-drift', 'morganchen1007/resnet-50-finetuned-resnet50_0831', 'eugenecamus/resnet-50-base-beans-demo', 'sallyanndelucia/resnet_weather_model'], '1': ['douwekiela/resnet-18-finetuned-dogfood', 'SiddharthaM/resnet-18-feature-extraction']}}, {'ResNet': ['microsoft/resnet-34', 'microsoft/resnet-152', 'microsoft/resnet-101', 'microsoft/resnet-18']})
```
