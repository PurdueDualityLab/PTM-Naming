
import torchview
import torch
from typing import Tuple, Any, List, Dict, Set
from torchview.computation_node.base_node import Node
from torchview.computation_node.compute_node import ModuleNode, TensorNode, FunctionNode
from torchview.computation_graph import ComputationGraph
import onnx
from onnx import NodeProto, GraphProto


# A class that stores a parameter-value pair

class ParamInfo():
    def __init__(
        self,
        param_name: str,
        param_value: Any = None
    ) -> None:
        self.param_name = param_name
        self.param_value = param_value
    
    def __eq__(self, other) -> bool:
        if self.param_name != other.param_name:
            return False
        return self.param_value == other.param_value
    
    def __str__(self) -> str:
        return '<' + self.param_name + ': ' + str(self.param_value) + '>'
    
    def __hash__(self) -> int:
        return hash(str(self.param_name) + str(self.param_value))


# A class that stores useful information of a node(layer)

class NodeInfo():

    UNUSED_PARAM_SET = {
        'training',
        'training_mode'
    }

    PARAMETERS_DEFAULT_ONNX = {
        'BatchNormalization': {
            'epsilon': [9.999999747378752e-06],
            'momentum': [0.8999999761581421],
            'spatial': [1]
            #'consumed_inputs': []
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

    def __init__(
        self,
        node_id: int = -1,
        input_shape: List[Tuple[int, ...]] = None,
        output_shape: List[Tuple[int, ...]] = None,
        operation: str = 'Undefined',
        parameters: List[ParamInfo] = None,
        is_input_node: bool = False,
        is_output_node: bool = False,
        sorting_identifier: str = None,
        sorting_hash: int = None,
        preorder_visited: bool = False,
        postorder_visited: bool = False # handles cyclic graphs
    ) -> None:
        self.node_id = node_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.operation = operation
        self.parameters = parameters
        self.is_input_node = is_input_node
        self.is_output_node = is_output_node
        self.sorting_identifier = sorting_identifier
        self.sorting_hash = sorting_hash
        self.preorder_visited = preorder_visited
        self.postorder_visited = postorder_visited

    def __hash__(self):
        return hash(self.node_id)

    # fill in the class var for module node type
    def fill_info_module_node(self, node: ModuleNode) -> None: 
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name
        self.parameters = []
        
        for attr_name, attr_val in node.module_unit.__dict__.items():
            if attr_name[0] != '_' and attr_name not in self.UNUSED_PARAM_SET and 'All' not in self.UNUSED_PARAM_SET: # include non-private attributes
                self.parameters.append(ParamInfo(attr_name, attr_val))

        if 'All' in self.UNUSED_PARAM_SET: self.parameters = None

    # fill in the class var for tensor node type
    def fill_info_tensor_node(self, node: TensorNode) -> None:
        self.node_id = node.node_id
        if node.name == 'auxiliary-tensor':
            self.is_input_node = True
            self.output_shape = node.tensor_shape
        if node.name == 'output-tensor':
            self.is_output_node = True
            self.input_shape = node.tensor_shape

    # fill in the class var for function node type
    def fill_info_function_node(self, node: FunctionNode) -> None:
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name

    def fill_info_from_onnx(
            self, 
            node: NodeProto = None, 
            input: List[Tuple[int, ...]] = None, 
            output: List[Tuple[int, ...]] = None,
            is_input: bool = False,
            is_output: bool = False,
            custom_id: int = -501
        ) -> None:
        self.node_id = custom_id
        self.input_shape = input
        self.output_shape = output
        self.is_input_node = is_input
        self.is_output_node = is_output
        if not (is_input or is_output):
            self.operation = node.op_type

            self.parameters = []

            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.FLOAT:
                    self.parameters.append(ParamInfo(attr.name, attr.f))
                elif attr.type == onnx.AttributeProto.INT:
                    self.parameters.append(ParamInfo(attr.name, attr.i))
                elif attr.type == onnx.AttributeProto.STRING:
                    self.parameters.append(ParamInfo(attr.name, attr.s))
                elif attr.type == onnx.AttributeProto.TENSOR:
                    self.parameters.append(ParamInfo(attr.name, attr.t))
                elif attr.type == onnx.AttributeProto.GRAPH:
                    self.parameters.append(ParamInfo(attr.name, attr.g))
                elif attr.type == onnx.AttributeProto.INTS:
                    self.parameters.append(ParamInfo(attr.name, tuple(attr.ints)))
                elif attr.type == onnx.AttributeProto.FLOATS:
                    self.parameters.append(ParamInfo(attr.name, tuple(attr.floats)))
                elif attr.type == onnx.AttributeProto.STRINGS:
                    self.parameters.append(ParamInfo(attr.name, tuple(attr.strings)))
                elif attr.type == onnx.AttributeProto.TENSORS:
                    self.parameters.append(ParamInfo(attr.name, tuple(attr.tensors)))
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    self.parameters.append(ParamInfo(attr.name, tuple(attr.graphs)))
                else:
                    self.parameters.append(ParamInfo(attr.name, None))
                
                if self.operation in self.PARAMETERS_DEFAULT_ONNX:
                    if attr.name in self.PARAMETERS_DEFAULT_ONNX[self.operation]:
                        if self.parameters[-1].param_value == self.PARAMETERS_DEFAULT_ONNX[self.operation][attr.name][0]:
                            self.parameters.pop()      
                if attr.name in self.UNUSED_PARAM_SET:
                    self.parameters.pop()

    def param_compare(self, other) -> bool:
        if self.parameters == None:
            return other.parameters == None
        for p in self.parameters:
            if p not in other.parameters and p not in self.UNUSED_PARAM_SET:
                return False
        return True

    # A list of equivalent utility function useful for comparison
    def __eq__(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation and \
            self.param_compare(other)
        )
    
    def op_eq(self, other) -> bool:
        return (
            self.operation == other.operation
        )
    
    def dim_eq(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation
        )
    
    def param_eq(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation and \
            self.param_compare(self, other)
        )
    
    # A helper function for filling class var by identifying the type of the inputted node
    def fill_info(self, node: Node) -> None:
        if type(node) == ModuleNode:
            self.fill_info_module_node(node)
        elif type(node) == TensorNode:
            self.fill_info_tensor_node(node)
        elif type(node) == FunctionNode:
            self.fill_info_function_node(node)

    # Generate the 'head' of the sorting identifier(sequence) for this node
    def generate_sorting_identifier_head(self) -> str:
        if self.is_input_node: return '[INPUT/{}]'.format(self.output_shape)
        if self.is_output_node: return '[OUTPUT/{}]'.format(self.input_shape)
        if self.parameters != None:
            pm_list = []
            for pm in self.parameters:
                pm_list.append(str(pm))
            return '[{}/{}/{}/{}]'.format(str(self.input_shape), str(self.output_shape), self.operation, str(pm_list))
        else:
            return '[{}/{}/{}]'.format(str(self.input_shape), str(self.output_shape), self.operation)
    
    def generate_sorting_hash(self) -> int:
        return hash(self.generate_sorting_identifier_head())

    # to string function
    def __str__(self) -> str:
        if self.is_input_node: return '[INPUT] out: {}'.format(self.output_shape)
        if self.is_output_node: return '[OUTPUT] in: {}'.format(self.input_shape)
        if self.parameters != None:
            pm_list = []
            for pm in self.parameters:
                pm_list.append(str(pm))
            return '[{}] in: {} out: {} {}'.format(self.operation, str(self.input_shape), str(self.output_shape), str(pm_list))
        else:
            return '[{}] in: {} out: {}'.format(self.operation, str(self.input_shape), str(self.output_shape))


# A class that stores all the mapping and some list/set of useful information

class Mapper():

    def __init__(
        self,
        node_info_obj_set: Set[NodeInfo] = None,
        node_id_to_node_obj_mapping: Dict[int, NodeInfo] = None,
        edge_node_info_list: List[Tuple[NodeInfo, NodeInfo]] = None
    ) -> None:
        self.node_info_obj_set = node_info_obj_set
        self.node_id_to_node_obj_mapping = node_id_to_node_obj_mapping
        self.edge_node_info_list = edge_node_info_list

    # populate the class var
    # node_info_obj_set: A set of all NodeInfo objects
    # node_id_to_node_obj_mapping: A map of [NodeInfo.node_id -> NodeInfo]
    # edge_node_info_list: A list of all the edges represented as Tuple[NodeInfo, NodeInfo], where the first NodeInfo points to the second
    def populate_class_var(
        self,
        edge_list: List[Tuple[Node, Node]]
    ) -> None:
        self.node_info_obj_set = set()
        self.node_id_to_node_obj_mapping = {}
        self.edge_node_info_list = []
        for edge_tuple in edge_list:
            
            if edge_tuple[0].node_id not in self.node_id_to_node_obj_mapping:
                n_info_0 = NodeInfo()
                n_info_0.fill_info(edge_tuple[0])
                self.node_info_obj_set.add(n_info_0)
                self.node_id_to_node_obj_mapping[n_info_0.node_id] = n_info_0
            else:
                n_info_0 = self.node_id_to_node_obj_mapping[edge_tuple[0].node_id]


            if edge_tuple[1].node_id not in self.node_id_to_node_obj_mapping:
                n_info_1 = NodeInfo()
                n_info_1.fill_info(edge_tuple[1])
                self.node_info_obj_set.add(n_info_1)
                self.node_id_to_node_obj_mapping[n_info_1.node_id] = n_info_1
            else:
                n_info_1 = self.node_id_to_node_obj_mapping[edge_tuple[1].node_id]

            self.edge_node_info_list.append((n_info_0, n_info_1))

    def populate_class_var_from_onnx(
        self,
        onnx_model: Any
    ):
        self.node_info_obj_set = set()
        self.node_id_to_node_obj_mapping = {}
        self.edge_node_info_list = []

        node_list = onnx_model.graph.node

        # create input name -> index map
        in2idx_map = {}

        for i in range(len(node_list)):
            for input in node_list[i].input:
                if input not in in2idx_map:
                    in2idx_map[input] = []
                in2idx_map[input].append(i)

        # create input name -> tensor shape map, this could be buggy
        '''
        inname2shape_map: dict = {
            input_tensor.name: [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            for input_tensor in onnx_model.graph.input
        }'''
        
        inname2shape_map: dict = {init.name: init.dims for init in onnx_model.graph.initializer}

        # Add input and output nodes
        # Assumes all node names are unique
        initializer_names = set(init.name for init in onnx_model.graph.initializer)
        actual_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initializer_names]

        input_nodes = actual_inputs
        output_nodes = onnx_model.graph.output
        
        input_node_info_list = []
        output_node_info_list = []
        io_id_cnt = -500 # use to calculate an id for an input/output NodeInfo obj
        # input node id = -500 + node index in input_nodes
        # output node id = -500 + len(input_nodes) + node index in output_nodes
        for input in input_nodes:
            input_node_info = NodeInfo()
            input_node_info.fill_info_from_onnx(custom_id=io_id_cnt, is_input=True)
            self.node_info_obj_set.add(input_node_info)
            self.node_id_to_node_obj_mapping[io_id_cnt] = input_node_info
            io_id_cnt += 1
            input_node_info_list.append(input_node_info)
        for output in output_nodes:
            output_node_info = NodeInfo()
            output_node_info.fill_info_from_onnx(custom_id=io_id_cnt, is_output=True)
            self.node_info_obj_set.add(output_node_info)
            self.node_id_to_node_obj_mapping[io_id_cnt] = output_node_info
            io_id_cnt += 1
            output_node_info_list.append(output_node_info)


        output_name_set = {output for node in node_list for output in node.output}
        input_nodes_name = [i.name for i in input_nodes]
        output_nodes_name = [o.name for o in output_nodes]

        # create index -> input tensor shape map
        idx2shape_map = {}

        for i in range(len(node_list)):
            shape_list = []
            for input in node_list[i].input:
                if input in inname2shape_map and input not in input_nodes_name:
                    shape_list.append(tuple(inname2shape_map[input]))

            idx2shape_map[i] = shape_list

        #for k, v in inname2shape_map.items():
        #    print(k, v)
        
        for input_name, indexes in in2idx_map.items():
            # omit unused inputs
            if input_name not in output_name_set and input_name not in input_nodes_name:
                continue
            for in_idx in indexes:
                # 
                for output_name in node_list[in_idx].output:
                    # successfully find a connection
                    if output_name in output_nodes_name or output_name in in2idx_map:
                        if output_name in in2idx_map:
                            out_indexes = in2idx_map[output_name]
                        else:
                            out_indexes = [-500 + len(input_nodes) + output_nodes_name.index(output_name)]

                        if input_name in input_nodes_name:
                            start_node_info = input_node_info_list[input_nodes_name.index(input_name)]

                            # Fix issue that this program omits the first layer
                            if in_idx in self.node_id_to_node_obj_mapping:
                                end_node_info = self.node_id_to_node_obj_mapping[in_idx]
                            else:
                                end_node_info = NodeInfo()
                                end_node_info.fill_info_from_onnx(
                                    node = node_list[in_idx],
                                    input = idx2shape_map[in_idx],
                                    custom_id = in_idx
                                )
                                self.node_id_to_node_obj_mapping[in_idx] = end_node_info
                                self.node_info_obj_set.add(end_node_info)
                            
                            self.edge_node_info_list.append((start_node_info, end_node_info))
                            start_node_info = end_node_info #???
                            #

                        elif in_idx in self.node_id_to_node_obj_mapping:
                            start_node_info = self.node_id_to_node_obj_mapping[in_idx]
                        else:
                            start_node_info = NodeInfo()
                            start_node_info.fill_info_from_onnx(
                                node = node_list[in_idx], 
                                input = idx2shape_map[in_idx],
                                custom_id = in_idx
                            )
                            self.node_id_to_node_obj_mapping[in_idx] = start_node_info
                            self.node_info_obj_set.add(start_node_info)
                        
                        for out_idx in out_indexes:

                            if out_idx in self.node_id_to_node_obj_mapping:
                                end_node_info = self.node_id_to_node_obj_mapping[out_idx]
                            else:
                                end_node_info = NodeInfo()
                                end_node_info.fill_info_from_onnx(
                                    node = node_list[out_idx],
                                    input = idx2shape_map[out_idx],
                                    custom_id = out_idx
                                )
                                self.node_id_to_node_obj_mapping[out_idx] = end_node_info
                                self.node_info_obj_set.add(end_node_info)
                            
                            self.edge_node_info_list.append((start_node_info, end_node_info))
        

    # returns an adjacency 'dictionary' that maps NodeInfo.node_id to a list of all the 'next node's it points to
    def get_adj_dict(self, options: Set = None) -> Dict[int, List[NodeInfo]]:
        adj_dict: Dict[int, List[NodeInfo]] = dict()
        for node_info_tuple in self.edge_node_info_list:
            if node_info_tuple[0].node_id not in adj_dict:
                adj_dict[node_info_tuple[0].node_id] = []
            adj_dict[node_info_tuple[0].node_id].append(node_info_tuple[1])
        
        if options != None and 'remove_identity' in options:
            for n_id, next_nodes in adj_dict.items():
                cleared = False
                while not cleared:
                    cleared = True
                    for next_node in next_nodes:
                        if next_node.operation == 'Identity':
                            cleared = False
                            adj_dict[n_id].remove(next_node)
                            for next_next_node in adj_dict[next_node.node_id]:
                                adj_dict[n_id].append(next_next_node)
                            adj_dict[next_node.node_id] = []


        return adj_dict


# A class that handles the majority of traversing process of the comparator

class Traverser():

    def __init__(
        self,
        mapper: Mapper,
        use_hash: bool = False
    ):
        self.mapper = mapper
        self.input_node_info_obj_list: List[TensorNode] = []
        self.output_node_info_obj_list: List[TensorNode] = []
        self.use_hash = use_hash
        for edge_node_info_tuple in mapper.edge_node_info_list: # identify input/output node and put them into the class var
            if edge_node_info_tuple[0].is_input_node and edge_node_info_tuple[0] not in self.input_node_info_obj_list:
                self.input_node_info_obj_list.append(edge_node_info_tuple[0])
            if edge_node_info_tuple[1].is_output_node and edge_node_info_tuple[1] not in self.output_node_info_obj_list:
                self.output_node_info_obj_list.append(edge_node_info_tuple[1])
        self.adj_dict: Dict[int, List[NodeInfo]] = mapper.get_adj_dict({'remove_identity'})

    # A helper function that helps to sort a list of node based on their sorting identifiers
    def sorted_node_info_list(self, node_info_list: List[NodeInfo]):
        if self.use_hash: return sorted(node_info_list, key=lambda obj: obj.sorting_hash)
        return sorted(node_info_list, key=lambda obj: obj.sorting_identifier)

    # A function that clears the visited class var for future traversals
    def reset_visited_field(self) -> None:
        node_info_obj_set = self.mapper.node_info_obj_set
        for node_info_obj in node_info_obj_set:
            node_info_obj.preorder_visited = False
            node_info_obj.postorder_visited = False

    
    # A function that assign sorting identifier to each of the NodeInfo in postorder
    def assign_sorting_identifier(self) -> None:

        self.reset_visited_field()
        
        def remove_common_suffix(strlist: List[str]) -> Tuple[List[str], str]:
            # If list is empty, return it as is
            if not strlist:
                return strlist, ""
                
            # Initialize the common suffix to the reversed first string in the list
            suffix = strlist[0][::-1]
            min_len = len(suffix)

            # Compare the reversed strings from left to right (original strings from right to left)
            for string in strlist[1:]:
                string = string[::-1]
                min_len = min(min_len, len(string))
                for i in range(min_len):
                    if string[i] != suffix[i]:
                        min_len = i
                        break
                # Truncate the common suffix to its common part
                suffix = suffix[:min_len]

            # Remove the common suffix from each string
            if min_len > 0:
                return [string[:-min_len] for string in strlist], suffix[::-1]
            else:
                return strlist, ""
            
        def traverse(curr_node_info_obj: NodeInfo) -> None:

            curr_node_info_obj.preorder_visited = True

            sorting_identifier, sorting_hash = None, None
            if self.use_hash: sorting_hash: int = curr_node_info_obj.generate_sorting_hash()
            else: sorting_identifier: str = curr_node_info_obj.generate_sorting_identifier_head()

            if curr_node_info_obj.node_id not in self.adj_dict: # output node
                
                if self.use_hash: curr_node_info_obj.sorting_hash = sorting_hash
                else: curr_node_info_obj.sorting_identifier = sorting_identifier

            else:

                for next_node_info_obj in self.adj_dict[curr_node_info_obj.node_id]:
                    if next_node_info_obj.preorder_visited or next_node_info_obj.postorder_visited: # handles cyclic graphs
                        continue
                    traverse(next_node_info_obj)

                sorted_next_obj_list = self.sorted_node_info_list(self.adj_dict[curr_node_info_obj.node_id]) # sort the next nodes


                if self.use_hash: # hash option
                    hash_sum = sorting_hash
                    for next_node_info_obj in sorted_next_obj_list:
                        hash_sum += next_node_info_obj.sorting_hash
                    curr_node_info_obj.sorting_hash = hash(hash_sum)
                    
                else:
                    sorting_identifier_list: List[str] = []

                    for next_node_info_obj in sorted_next_obj_list:
                        sorting_identifier_list.append(next_node_info_obj.sorting_identifier)

                    sorting_identifier_list, suffix = remove_common_suffix(sorting_identifier_list)

                    for s in sorting_identifier_list:
                        sorting_identifier += s
                    
                    sorting_identifier += suffix

                    curr_node_info_obj.sorting_identifier = sorting_identifier

            curr_node_info_obj.postorder_visited = True

            return

        for input_node_info_obj in self.input_node_info_obj_list: # Do the same traversal for all the inputs
            traverse(input_node_info_obj)

    

    # A function that generates a list of graph nodes based on the order of the sorting identifier
    # similar to the above func
    def generate_ordered_layer_list(self) -> List[NodeInfo]:

        self.assign_sorting_identifier()
        self.reset_visited_field()

        sorted_inputs: List[NodeInfo] = self.sorted_node_info_list(self.input_node_info_obj_list)

        ordered_layer_list: List[NodeInfo] = []

        def traverse(curr_node_info_obj: NodeInfo) -> None:

            curr_node_info_obj.preorder_visited = True

            ordered_layer_list.append(curr_node_info_obj)

            if curr_node_info_obj.node_id not in self.adj_dict:
                return
            
            next_obj_list = self.sorted_node_info_list(self.adj_dict[curr_node_info_obj.node_id])

            for next_node_info_obj in next_obj_list:

                if next_node_info_obj.preorder_visited or next_node_info_obj.postorder_visited:
                    continue

                traverse(next_node_info_obj)
                next_node_info_obj.postorder_visited = True

            return

        for input in sorted_inputs:
            traverse(input)
        
        return ordered_layer_list


# A helper function that combines all the process into one
def generate_ordered_layer_list_from_pytorch_model(
        model: Any, 
        inputs: Tuple[torch.Tensor], 
        graph_name: str = 'Untitled',
        depth: int = 16,
        use_hash: bool = False
    ) -> List[NodeInfo]:
    
    model_graph: ComputationGraph = torchview.draw_graph(
        model, inputs,
        graph_name=graph_name,
        depth=depth, 
        expand_nested=True
    )
    
    mapper = Mapper()
    mapper.populate_class_var(model_graph.edge_list)

    print('Mapper Populated')

    traverser = Traverser(mapper, use_hash)
    l = traverser.generate_ordered_layer_list()
    print('Ordered List Generated')
    return l

def generate_ordered_layer_list_from_onnx_model(
        model: Any,
        use_hash: bool = False
    ) -> List[NodeInfo]:
    mapper = Mapper()
    mapper.populate_class_var_from_onnx(model)

    traverser = Traverser(mapper, use_hash)
    l = traverser.generate_ordered_layer_list()
    return l

def generate_ordered_layer_list_from_onnx_model_with_id_and_connection(
        model: Any, 
        use_hash: bool = False
    ) -> List[Tuple[int, List[int]]]:
    
    mapper = Mapper()
    mapper.populate_class_var_from_onnx(model)

    #print('Mapper Populated')

    traverser = Traverser(mapper, use_hash)
    l = traverser.generate_ordered_layer_list()

    #print('Ordered List Generated')
    
    layer_id_connection_list: List[Tuple[int, List[int]]] = []

    for layer in l:
        connection_list = traverser.adj_dict[layer.node_id] if layer.node_id in traverser.adj_dict else []
        connection_id_list = []
        for node in connection_list:
            connection_id_list.append(node.node_id)
        layer_id_connection_list.append((layer.node_id, connection_id_list))
    return l, layer_id_connection_list

def generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(
        model: Any, 
        inputs: Tuple[torch.Tensor], 
        graph_name: str = 'Untitled',
        depth: int = 16,
        use_hash: bool = False
    ) -> List[Tuple[int, List[int]]]:
    
    print('Preparing tracing')

    model_graph: ComputationGraph = torchview.draw_graph(
        model, inputs,
        graph_name=graph_name,
        depth=depth, 
        expand_nested=True
    )

    print('Tracing successful')
    
    mapper = Mapper()
    mapper.populate_class_var(model_graph.edge_list)

    print('Mapper Populated')

    traverser = Traverser(mapper, use_hash)
    l = traverser.generate_ordered_layer_list()

    print('Ordered List Generated')
    
    layer_id_connection_list: List[Tuple[int, List[int]]] = []

    for layer in l:
        connection_list = traverser.adj_dict[layer.node_id] if layer.node_id in traverser.adj_dict else []
        connection_id_list = []
        for node in connection_list:
            connection_id_list.append(node.node_id)
        layer_id_connection_list.append((layer.node_id, connection_id_list))
    return l, layer_id_connection_list
        
# Adding a class var to torchview Node classes so the original Tensor/Module can be accessed
def patch():
    def new_tn_init(
            self,
            tensor,
            depth,
            parents=None,
            children=None,
            name='tensor',
            context=None,
            is_aux=False,
            main_node=None,
            parent_hierarchy=None,
        ):
        
        old_tn_init(
            self, tensor, depth, parents, children, name, context,
            is_aux, main_node, parent_hierarchy
        )
        
        self.tensor = tensor

    old_tn_init = torchview.computation_node.TensorNode.__init__

    torchview.computation_node.TensorNode.__init__ = new_tn_init

    def new_mn_init(
            self,
            module_unit,
            depth,
            parents = None,
            children = None,
            name = 'module-node',
            output_nodes = None,
        ):
        old_mn_init(self, module_unit, depth, parents, children, name, output_nodes)
        self.module_unit = module_unit

    old_mn_init = torchview.computation_node.ModuleNode.__init__

    torchview.computation_node.ModuleNode.__init__ = new_mn_init

def print_list(l):
    for i in l: print(i)