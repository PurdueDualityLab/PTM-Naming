import json
from utils import NodeInfo, ParamInfo
from typing import List, Tuple
from transformers import ResNetForImageClassification, AlbertForMaskedLM
from transformers import AutoModel, AutoTokenizer
from list_gen import OrderedListGenerator

def node_list_to_json(layer_list: List[NodeInfo], connection_info: List[Tuple[int, List[int]]], output_dir):
    if len(layer_list) != len(connection_info): 
        print('Unrecognized connection info')
        return None
    json_compat_list = []
    for i in range(len(layer_list)):
        node_dict = dict()
        curr_node = layer_list[i]

        node_dict['operation'] = curr_node.operation
        node_dict['input_shape'] = curr_node.input_shape
        node_dict['output_shape'] = curr_node.output_shape
        node_dict['node_id'] = curr_node.node_id

        if curr_node.node_id != connection_info[i][0]:
            print('Invalid connection info')
            return None
        
        node_dict['connects_to'] = connection_info[i][1]

        if curr_node.is_input_node:
            node_dict['type'] = 'Input'
        elif curr_node.is_output_node:
            node_dict['type'] = 'Output'
        else:
            node_dict['type'] = 'Middle'
        
        if curr_node.parameters == None:
            node_dict['parameters'] = None
        else:
            param_dict = dict()
            for p_i in curr_node.parameters:
                if type(p_i.param_value) not in [str, list, dict, tuple, int, float, True, False, None]:
                    param_dict[p_i.param_name] = str(p_i.param_value)
                else:
                    param_dict[p_i.param_name] = p_i.param_value
            node_dict['parameters'] = param_dict
        json_compat_list.append(node_dict)
    
    with open(output_dir, 'w') as f:
        json.dump(json_compat_list, f)

def read_node_list_from_json(json_dir):
    with open(json_dir) as f:
        data = json.load(f)
    l_l = []
    c_i = []
    for node in data:
        new_node_info = NodeInfo()
        new_node_info.operation = node['operation']
        new_node_info.input_shape = node['input_shape']
        new_node_info.output_shape = node['output_shape']
        new_node_info.node_id = node['node_id']
        if node['parameters'] != None:
            params = []
            for k, v in node['parameters'].items():
                params.append(ParamInfo(k, v))
            new_node_info.parameters = params
        new_node_info.is_input_node = node['type'] == 'Input'
        new_node_info.is_output_node = node['type'] == 'Output'
        c_i.append((node['node_id'], node['connects_to']))
        l_l.append(new_node_info)
    return l_l, c_i

def Custom_Test():
    t = AutoTokenizer.from_pretrained("bert-base-cased")
    m = AutoModel.from_pretrained('bert-base-cased')
    i = t("Test Input", return_tensors="pt")
    gen = OrderedListGenerator(m, i)
    l_l, c_i = gen.get_connection()
    node_list_to_json(l_l, c_i, '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/convert_test.json')
    ll2, ci2 = read_node_list_from_json('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/convert_test.json')
    node_list_to_json(ll2, ci2, '/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/convert_test2.json')

#Custom_Test()