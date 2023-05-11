from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
import onnx
from GetOnnxFileFromURL import getModel

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model3 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/resnet18-v2-7.onnx')

graph = model3.graph

initializers = {init.name: init for init in graph.initializer}

shapes = {vi.name: [dim.dim_value for dim in vi.type.tensor_type.shape.dim] for vi in graph.value_info}

prevname_bn = False

for i, node in enumerate(graph.node):
    for name in node.input:
        if name in initializers:
            tensor = initializers[name]
            if not ('batchnorm' in name and 'gamma' not in name and 'beta' not in name):
                print(name, tensor.dims)


np1 = list(model1.named_parameters())
np2 = list(model2.named_parameters())

for i in range(len(np1)):
    print(np1[i][0], np2[i][0])
    print(np1[i][1].size(), np2[i][1].size())

