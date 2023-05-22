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
import cProfile

PARAMETERS_DEFAULT = {
    'BatchNormalization': {
        'epsilon': [9.999999747378752e-06],
        'momentum': [0.8999999761581421],
        'spatial': [1],
        'consumed_inputs': []
    },
    'MaxPool': {
        'auto_pad': ["NOTSET"],
        'ceil_mode': [0],
        'dilation': [1],
        'strides': [1]
    },
    'Gemm': {
        'alpha': [1.0],
        'beta': [1.0],
        'transA': [0],
        'transB': [0]
    },
    'Flatten': {
        'axis': [1]
    }
}

UNUSED_ATTRIBUTE_NAME = {
    'training_mode'
}

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
#odel1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

#model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
model3 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/resnet18-v1-7.onnx')
'''
dummy_input = torch.randn(1, 3, 224, 224)
input_name, output_name = ['input'], ['output']
torch.onnx.export(
    model2, 
    dummy_input, 
    "/depot/davisjam/data/chingwo/PTM-Naming/pytorch-resnet34.onnx", 
    verbose=False, 
    input_names=input_name, 
    output_names=output_name, 
    do_constant_folding=False, 
    opset_version=16,
    training=torch.onnx.TrainingMode.TRAINING
    )'''
model4 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/temp.onnx')
model5 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/pytorch-resnet34.onnx')
model6 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/resnet34-v1-7.onnx')

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
                if attribute.name in UNUSED_ATTRIBUTE_NAME:
                    continue
                
                if attribute.type == onnx.AttributeProto.INT:
                    attributes[node_index][attribute.name] = [attribute.i]
                elif attribute.type == onnx.AttributeProto.INTS:
                    attributes[node_index][attribute.name] = attribute.ints
                elif attribute.type == onnx.AttributeProto.FLOAT:
                    attributes[node_index][attribute.name] = [attribute.f]
                elif attribute.type == onnx.AttributeProto.FLOATS:
                    attributes[node_index][attribute.name] = attribute.floats

                # Remove the attribute if it is in default value
                if name2node_mapping[node_name].op_type in PARAMETERS_DEFAULT:
                    if attribute.name in PARAMETERS_DEFAULT[name2node_mapping[node_name].op_type]:
                        if PARAMETERS_DEFAULT[name2node_mapping[node_name].op_type][attribute.name] == attributes[node_index][attribute.name]:
                            del attributes[node_index][attribute.name]

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
    
    diff_params: Iterator[str] = differ.compare(
        [str(p) for p in info_dict1["params"]], 
        [str(p) for p in info_dict2["params"]]
    )
    return {"op_types": diff_op_types, "params": diff_params, "dims": diff_dims}

def printLayerDiff(diff_dict: dict) -> None:
    print('op type difference')
    for item in diff_dict["op_types"]:
        print(item)
    
    print('\nparams difference') 
    for item in diff_dict["params"]:
        print(item)
    
    print('\ndims difference')
    for item in diff_dict["dims"]:
        print(item)

def printModelLayerDiff(model1: onnx.ModelProto, model2: onnx.ModelProto) -> None:
    printLayerDiff(
        calcLayerDiff(
        getLayerInfo(model1, sortLayerList(traverseGraph(model1), getSortingSequence(model1, getAdjList(model1)['adj_list']))), 
        getLayerInfo(model2, sortLayerList(traverseGraph(model2), getSortingSequence(model2, getAdjList(model2)['adj_list'])))))

def traverseGraph(model: onnx.ModelProto) -> list:
    graph: onnx.GraphProto = model.graph
    name2node_map: dict = {node.name: node for node in graph.node}
    in2node_map: dict = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name not in in2node_map: in2node_map[input_name] = []
            in2node_map[input_name] += [node]

    depth_mapping: dict = {node.name: -1 for node in graph.node}
    depth_mapping[graph.node[0].name] = 0

    def getDepthMap(node: onnx.NodeProto, depth: dict, i2n_map: dict, curr_depth: int) -> dict:
        if depth[node.name] < curr_depth:
            depth[node.name] = curr_depth
        for output_name in node.output:
            if output_name in i2n_map:
                for next_node in in2node_map[output_name]:
                    depth = getDepthMap(next_node, depth, i2n_map, curr_depth + 1)
        return depth
    
    dm = getDepthMap(graph.node[0], depth_mapping, in2node_map, 0)

    layer = [None] * (max(dm.values()) + 1)

    for k in dm.keys():
        if layer[dm[k]] == None: layer[dm[k]] = []
        layer[dm[k]] += [k]
    
    return layer

def getAdjList(model: onnx.ModelProto) -> dict:
    graph: onnx.GraphProto = model.graph
    name2node_map: dict = {node.name: node for node in graph.node}
    in2name_map: dict = {}
    for node in graph.node:
        for input in node.input:
            if input not in in2name_map: in2name_map[input] = []
            in2name_map[input] += [node.name]
    adj_list: dict = {}
    for node in graph.node:
        adj_list[node.name] = []
        for output in node.output:
            if output in in2name_map:
                for next_node_name in in2name_map[output]:
                    adj_list[node.name].append(next_node_name)
    return {'name2node_map': name2node_map, 'adj_list': adj_list}

def getDims(model: onnx.ModelProto, node: onnx.NodeProto, name2dim_map: dict) -> list:
    dims_list: list = []
    for input_param in node.input:
        if input_param in name2dim_map:
            dims_list.append(tuple(name2dim_map[input_param]))
    return dims_list

def getSortingSequence(model: onnx.ModelProto, adj_list: dict) -> dict:
    sorting_sequence_dict: dict = {}
    graph: onnx.GraphProto = model.graph
    name2node_map: dict = {node.name: node for node in graph.node}
    name2dim_map: dict = {init.name: init.dims for init in model.graph.initializer}

    def traverse(model: onnx.ModelProto, node_name: str, adj_list: dict, seq_dict: dict, n2n_map: dict, n2d_map: dict) -> dict:
        temp_list: list = []
        node: onnx.NodeProto = n2n_map[node_name]
        seq_dict[node_name]: str = '[{},{}]'.format(node.op_type, str(getDims(model, node, name2dim_map)))
        for next_node_name in adj_list[node_name]:
            traverse(model, next_node_name, adj_list, seq_dict, n2n_map, n2d_map)
            temp_list.append(seq_dict[next_node_name])
        temp_list = sorted(temp_list)
        for seq in temp_list:
            seq_dict[node_name] += seq
        return seq_dict

    sorting_sequence_dict = traverse(model, graph.node[0].name, adj_list, sorting_sequence_dict, name2node_map, name2dim_map)

    return sorting_sequence_dict

def sortLayerList(layer_list: list, seq_dict: dict) -> list:
    for idx in range(len(layer_list)):
        if len(layer_list[idx]) > 1:
            layer_list[idx] = sorted(layer_list[idx], key=lambda branch: seq_dict[branch])
    return layer_list

def analyzeModelLayerDiff(model1: onnx.ModelProto, model2: onnx.ModelProto) -> None:
    diff_dict: dict = calcLayerDiff(
        getLayerInfo(model1, sortLayerList(traverseGraph(model1), getSortingSequence(model1, getAdjList(model1)['adj_list']))), 
        getLayerInfo(model2, sortLayerList(traverseGraph(model2), getSortingSequence(model2, getAdjList(model2)['adj_list']))))
    op_type_different: bool = False
    dims_different: bool = False
    params_different: bool = False
    
    op_types_diff: list = []
    for s in diff_dict['op_types']: op_types_diff.append(s)
    dims_diff: list = []
    for s in diff_dict['dims']: dims_diff.append(s)
    params_diff: list = []
    for s in diff_dict['params']: params_diff.append(s)

    for str_ in op_types_diff:
        if str_[0] in ['-', '+']:
            op_type_different = True
            break
    for str_ in params_diff:
        if str_[0] in ['-', '+']:
            params_different = True
            break
    for str_ in dims_diff:
        if str_[0] in ['-', '+']:
            dims_different = True
            break

    if op_type_different: # need test
        print('Found differences in op types:')
        for s in op_types_diff:
            print(s)
    elif dims_different: # need test
        print('Found differences in dimensions')
        j: int = 0
        for i in range(len(op_types_diff)):
            print('[{}] {} {}'.format(i, op_types_diff[i], dims_diff[j]))
            if dims_diff[j][0] == '-':
                while dims_diff[j][0] != '+':
                    j += 1
                print('[{}] {} {}'.format(i, op_types_diff[i], dims_diff[j]))
                j += 1
                if j < len(dims_diff):
                    while dims_diff[j][2] != '{':
                        j += 1
                j -= 1
            j += 1
    elif params_different:
        print('Found differences in parameters:')
        j: int = 0
        for i in range(len(op_types_diff)):
            print('[{}] {} {} {}'.format(i, op_types_diff[i], dims_diff[i], params_diff[j]))
            if params_diff[j][0] == '-':
                while params_diff[j][0] != '+':
                    j += 1
                print('[{}] {} {} {}'.format(i, op_types_diff[i], dims_diff[i], params_diff[j]))
                j += 1
                if j < len(params_diff):
                    while params_diff[j][2] != '{':
                        j += 1
                j -= 1
            j += 1
    else:
        print("Found no difference.")

def main():
    analyzeModelLayerDiff(model3, model4)
    analyzeModelLayerDiff(model6, model5)
    pass

cProfile.run('main()')

