import networkx
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

from utils import NodeInfo, ParamInfo, Mapper, patch

patch()

model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

inputs = torch.randn(1, 3, 224, 224)

model_graph = torchview.draw_graph(
        model, inputs,
        graph_name='microsoft/resnet',
        depth=6, #default is 3
        # get the nested layers
        expand_nested=True
    )

mapper = Mapper()
mapper.populate_class_var(model_graph.edge_list)
print(mapper.node_id_to_node_obj_mapping)


