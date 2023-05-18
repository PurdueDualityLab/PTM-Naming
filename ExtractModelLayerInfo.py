from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
import onnx
from onnx import shape_inference
from GetOnnxFileFromURL import getModel
from torch.onnx.verification import find_mismatch
import difflib
from typing import Iterator
import onnxoptimizer

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#odel1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model3 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/resnet18-v1-7.onnx')

dummy_input = torch.randn(1, 3, 224, 224)
input_name, output_name = ['input'], ['output']
torch.onnx.export(
    model2, 
    dummy_input, 
    "/depot/davisjam/data/chingwo/PTM-Naming/temp.onnx", 
    verbose=True, 
    input_names=input_name, 
    output_names=output_name, 
    do_constant_folding=False, 
    opset_version=16,
    training=torch.onnx.TrainingMode.TRAINING
    )
model4 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/temp.onnx')

def getLayerInfo(model: onnx.ModelProto, layer: list) -> dict:
    name2node_mapping: dict = {node.name: node for node in model.graph.node}
    initializers: dict = {init.name: init for init in model.graph.initializer}
    op_types: list = []
    attributes: list = []
    dims: list = []
    nodes: onnx.NodeProto = model.graph.node
    node_index = 0
    for layer_depth in range(len(layer)):
        for node_name in layer[layer_depth]:
            op_types.append(name2node_mapping[node_name].op_type)
            attributes.append(dict())
            dims.append(list())
            for attribute in name2node_mapping[node_name].attribute:
                if attribute.type == onnx.AttributeProto.INT:
                    attributes[node_index][attribute.name] = [attribute.i]
                elif attribute.type == onnx.AttributeProto.INTS:
                    attributes[node_index][attribute.name] = attribute.ints
                elif attribute.type == onnx.AttributeProto.FLOAT:
                    attributes[node_index][attribute.name] = [attribute.f]
                elif attribute.type == onnx.AttributeProto.FLOATS:
                    attributes[node_index][attribute.name] = attribute.floats
            for name in name2node_mapping[node_name].input:
                if name in initializers:
                    dims[node_index].append(tuple(initializers[name].dims))
            node_index += 1
    return {"op_types": op_types, "params": attributes, "dims": dims}

def printLayerInfo(info_dict: dict) -> None:
    node_count: int = len(info_dict["op_types"])
    for node_index in range(node_count):
        print(info_dict["op_types"][node_index], info_dict["dims"][node_index])
        print(info_dict["params"][node_index])

def calcLayerDiff(info_dict1: dict, info_dict2: dict) -> dict:
    differ: difflib.Differ = difflib.Differ()
    diff_op_types: Iterator[str] = differ.compare(info_dict1["op_types"], info_dict2["op_types"])
    diff_dims: Iterator[str] = differ.compare([str(i) for i in info_dict1["dims"]], [str(i) for i in info_dict2["dims"]])
    diff_params: Iterator[str] = differ.compare(info_dict1["params"], info_dict2["params"])
    return {"op_types": diff_op_types, "params": diff_params, "dims": diff_dims}

def printLayerDiff(diff_dict: dict) -> None:
    print('op type difference')
    for item in diff_dict["op_types"]:
        print(item)
    '''
    print('\nparams difference') TODO: write a difference checker for dict type
    for item in diff_dict["params"]:
        print(item)
    '''
    print('\ndims difference')
    for item in diff_dict["dims"]:
        print(item)

def printModelLayerDiff(model1: onnx.ModelProto, model2: onnx.ModelProto) -> None:
    printLayerDiff(calcLayerDiff(getLayerInfo(model1, traverseGraph(model1)), getLayerInfo(model2, traverseGraph(model2))))

def traverseGraph(model: onnx.ModelProto) -> list: # TODO: handle the case: multiple outputs in a node
    graph: onnx.GraphProto = model.graph
    out2in_mapping: dict = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in out2in_mapping:
                out2in_mapping[input_name] = []
            out2in_mapping[input_name] += node.output
    name2node_mapping: dict = {node.name: node for node in graph.node}

    depth_mapping: dict = {node.name: -1 for node in graph.node}
    depth_mapping[graph.node[0].name] = 0

    def getDepthMap(node: onnx.NodeProto, depth: dict, input_map: dict, name_map: dict, curr_depth: int) -> dict:
        if depth[node.name] < curr_depth:
            depth[node.name] = curr_depth
        if node.name in input_map:
            for output_name in input_map[node.name]:
                depth = getDepthMap(name_map[output_name], depth, input_map, name_map, curr_depth + 1)
        return depth
    
    dm = getDepthMap(graph.node[0], depth_mapping, out2in_mapping, name2node_mapping, 0)
    layer = [None] * (max(dm.values()) + 1)

    for k, v in out2in_mapping.items():
        print(k, v)

    for k in dm.keys():
        if layer[dm[k]] == None: layer[dm[k]] = []
        layer[dm[k]] += [k]
    for i in range(len(layer)):
        layer[i] = sorted(layer[i], key=lambda node_name: name2node_mapping[node_name].op_type)

    for l in layer:
        print(l)
    
    return layer
'''
print(onnx.helper.printable_graph(model4.graph))
l = traverseGraph(model4)
i_d = getLayerInfo(model4, l)
printLayerInfo(i_d)
'''
for i in range(20):
    print(model4.graph.node[i])


