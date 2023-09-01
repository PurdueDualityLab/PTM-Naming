
from utils import (
    NodeInfo, 
    generate_ordered_layer_list_from_pytorch_model, 
    patch, 
    generate_ordered_layer_list_from_pytorch_model_with_id_and_connection,
    generate_ordered_layer_list_from_onnx_model_with_id_and_connection,
    generate_ordered_layer_list_from_onnx_model
)
from typing import Any, Tuple, List
from torch import Tensor

class OrderedListGenerator():

    def __init__(
        self,
        model: Any,
        inputs: Tuple[Tensor, ...] = None,
        mode: str = 'pytorch',
        use_hash: bool = False
    ) -> None:
        """
        OrderedListGenerator constructor function

        Parameters:
        model (Any): Any model type from pytorch or onnx
        inputs (tuple): Input of the model for tracing, not necessary for
        onnx model
        mode (str): Specify the list generator on generating list for onnx
        or pytorch model
        use_hash (bool): Use hash while assigning layer identifiers, set
        it to True for increase performance

        Returns:
        None
        """
        self.model = model
        self.inputs = inputs
        self.mode = mode
        self.use_hash = use_hash
        patch()

    def get_ordered_list(self) -> List[NodeInfo]:
        """
        Returns an ordered list for the model

        Parameters:
        None

        Returns:
        list: A list of NodeInfo objects which contains all the information
        of a layer in the model
        """
        if self.mode == 'pytorch':
            return generate_ordered_layer_list_from_pytorch_model(self.model, self.inputs, use_hash=self.use_hash)
        if self.mode == 'onnx':
            return generate_ordered_layer_list_from_onnx_model(self.model, use_hash=self.use_hash)


    def print_ordered_list(self) -> None:
        """
        Print an ordered list for the model

        Parameters:
        None

        Returns:
        None
        """
        if self.mode == 'pytorch':
            ordered_list = generate_ordered_layer_list_from_pytorch_model(self.model, self.inputs, use_hash=self.use_hash)
        if self.mode == 'onnx':
            ordered_list = generate_ordered_layer_list_from_onnx_model(self.model, use_hash=self.use_hash)
        for layer_node in ordered_list:
            print(layer_node)

    def get_connection(self) -> List[Tuple[int, List[int]]]:
        """
        Return an ordered list for the model as well as the connection information,
        essentially making the ordered list an adjacency list.

        Parameters:
        None

        Returns:
        list: An ordered list with connection information
        """
        if self.mode == 'pytorch':
            l = generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(self.model, self.inputs, use_hash=self.use_hash)
        if self.mode == 'onnx':
            l = generate_ordered_layer_list_from_onnx_model_with_id_and_connection(self.model, use_hash=self.use_hash)
        return l[0], l[1]
    
    def print_connection(self) -> None:
        """
        Print an ordered list for the model as well as the connection information,
        essentially making the ordered list an adjacency list

        Parameters:
        None

        Returns:
        None
        """
        l = generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(self.model, self.inputs)
        for layer_node, connection_info in zip(l[0], l[1]):
            print('[{}] {} -> {}'.format(connection_info[0], layer_node, connection_info[1]))


    def vectorize(self):
        """
        Vectorize the model using an n-gram like approach

        Unused Function
        """
        l_l, c_i = self.get_connection()        

        def get_freq_vec_l(layer_list, connection_info):
            freq_vec = dict()
            id_to_node_map = dict()
            for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
                id_to_node_map[layer_connection_info[0]] = layer_node
                
            def make_node_string(n: NodeInfo):
                if n.is_input_node:
                    return '[INPUT]'
                if n.is_output_node:
                    return '[OUTPUT]'
                return '{} {}->{}'.format(n.operation, n.input_shape, n.output_shape)

            for layer_node, layer_connection_info in zip(layer_list, connection_info):
                curr_node_str = make_node_string(layer_node)
                for next_layer_id in layer_connection_info[1]:
                    next_node_str = make_node_string(id_to_node_map[next_layer_id])
                    combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                    if combined_str not in freq_vec:
                        freq_vec[combined_str] = 0
                    freq_vec[combined_str] += 1
            
            return freq_vec
        
        def get_freq_vec_p(layer_list: List[NodeInfo]):
            freq_vec = dict()
            for l in layer_list:
                p_str_list = []
                if l.parameters != None:
                    for p in l.parameters:
                        p_str_list.append('<{}, {}>'.format(p.param_name, p.param_value))
                if l.is_input_node:
                    l_str = '[INPUT]'
                elif l.is_output_node:
                    l_str = '[OUTPUT]'
                else:
                    l_str = '{} {}'.format(l.operation, p_str_list if len(p_str_list) else '')
                if l_str not in freq_vec:
                    freq_vec[l_str] = 0
                freq_vec[l_str] += 1
            return freq_vec
        
        def get_freq_vec_pl(layer_list, connection_info):
            freq_vec = dict()
            id_to_node_map = dict()
            for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
                id_to_node_map[layer_connection_info[0]] = layer_node
                
            def make_node_string(n: NodeInfo):
                if n.is_input_node:
                    return '[INPUT]'
                if n.is_output_node:
                    return '[OUTPUT]'
                return n.operation

            for layer_node, layer_connection_info in zip(layer_list, connection_info):
                curr_node_str = make_node_string(layer_node)
                for next_layer_id in layer_connection_info[1]:
                    next_node_str = make_node_string(id_to_node_map[next_layer_id])
                    combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                    if combined_str not in freq_vec:
                        freq_vec[combined_str] = 0
                    freq_vec[combined_str] += 1
            
            return freq_vec
        
        def get_freq_vec_d(layer_list):
            freq_vec_d, freq_vec_dn = dict(), dict()
            for l in layer_list:
                d_list = []
                if l.input_shape != None:
                    for s_in in l.input_shape: 
                        d_list.append(s_in)
                if l.output_shape != None:
                    for s_out in l.output_shape: 
                        d_list.append(s_out)
                for t in d_list:
                    if str(t) not in freq_vec_d:
                        freq_vec_d[str(t)] = 0
                    freq_vec_d[str(t)] += 1
                    for n in t:
                        if n not in freq_vec_dn:
                            freq_vec_dn[n] = 0
                        freq_vec_dn[n] += 1
                
            return freq_vec_d, freq_vec_dn
        
        fv_l = get_freq_vec_l(l_l, c_i)
        fv_p = get_freq_vec_p(l_l)
        fv_pl = get_freq_vec_pl(l_l, c_i)
        fv_d, fv_dn = get_freq_vec_d(l_l)

        return fv_l, fv_p, fv_pl, fv_d, fv_dn
