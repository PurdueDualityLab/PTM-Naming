import networkx
from torch import nn
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from torch.autograd import Variable
from datasets import load_dataset
import time

# module use /depot/davisjam/data/chingwo/general_env/modules
# module load conda-env/general_env-py3.8.5


import graphviz

# when running on VSCode run the below command
# svg format on vscode does not give desired result
graphviz.set_jupyter_format('png')

import matplotlib.pyplot as plt
import torchview

from utils import NodeInfo, ParamInfo, Mapper, patch, generate_ordered_layer_list_from_pytorch_model

patch()

class MultiInputModel(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(MultiInputModel, self).__init__()
        self.layer1 = nn.Linear(input_dim1, output_dim)
        self.layer2 = nn.Linear(input_dim2, output_dim)

    def forward(self, input1, input2):
        output1 = self.layer1(input1)
        output2 = self.layer2(input2)
        return output1 + output2

# Create model with dimensions of inputs and output
model = MultiInputModel(10, 20, 1)

# Example inputs
input1 = torch.randn(32, 10)  # Batch of 32, each with 10 features
input2 = torch.randn(32, 20)  # Batch of 32, each with 20 features

# Forward pass
inputs = (input1, input2)

model_graph = torchview.draw_graph(
        model, inputs,
        graph_name='t',
        depth=6, 
        expand_nested=True
    )

print(model_graph.edge_list)
for edge_tuple in model_graph.edge_list:
    n_info_0 = NodeInfo()
    n_info_1 = NodeInfo()
    n_info_0.fill_info(edge_tuple[0])
    n_info_1.fill_info(edge_tuple[1])
    print(n_info_0, n_info_1)

# TODO: need to handle function node

