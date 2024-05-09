"""
This module contains the Node class, which is used to represent the nodes in the computation graph.
"""
from typing import List, Tuple
import torch
from torch import Tensor
from ANN.ann_layer import AbstractNNLayer
from ANN.ann_layer_param import AbstractNNLayerParam

SKIP_LAYER_PARAM = {
    'bias', 'T_destination', 'call_super_init', 'training', 'dump_patches', 
    'running_var', 'running_mean', 'num_batches_tracked', 'track_running_stats',
}

class Node:
    """
    The Node class is used to represent the nodes in the computation graph.

    Attributes:
        id: A unique identifier for the node.
        parents: A list of parent nodes.
        children: A list of child nodes.
    """
    _next_id = 0

    def __init__(self):
        self.id = Node._next_id
        Node._next_id += 1
        self.parents = []
        self.children = []

    def add_child(self, child):
        """
        This method adds a child node to the current node.

        Args:
            child: The child node to be added.
        """
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

    def __repr__(self):
        return f'<Node [{self.id}]> -> {[child.id for child in self.children]}'
    
class TensorNode(Node):
    """
    The TensorNode class is used to represent tensor nodes in the computation graph.

    Attributes:
        tensor: The tensor object associated with the node.
        shape: The shape of the tensor.
    """
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.tensor = tensor
        self.shape = tensor.shape

    def __repr__(self):
        return f'<TensorNode [{self.id}] shape={self.shape}> -> {[child.id for child in self.children]}'

class FunctionNode(Node):
    """
    The FunctionNode class is used to represent function nodes (PyTorch Function/Module) in the 
    computation graph.

    Attributes:
        operation: The operation performed by the function.
        contained_in_module: A flag indicating whether the function is contained in a module.
        module_info: Information about the module containing the function.
    """
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.contained_in_module = False
        self.module_info = None

    def to_annlayer(self, get_weight=False) -> Tuple[AbstractNNLayer, Tuple[List[int], List[int]]]:
        """
        Convert the FunctionNode to an AbstractNNLayer object.

        Args:
            get_weight: A flag indicating whether to get the weights of the layer.

        Returns:
            An AbstractNNLayer object representing the FunctionNode.
            A tuple with the input and output IDs.
        """
        
        # merge parent and children TensorNode objects into input and output tensors
        input_tensors = [parent.tensor for parent in self.parents if isinstance(parent, TensorNode)]
        output_tensors = [child.tensor for child in self.children if isinstance(child, TensorNode)]

        # handles ID when merging, get the input/output IDs with
        # the ID from parent of input_tensors and children of output_tensors
        input_ids = []
        output_ids = []
        for inp_node in self.parents:
            if isinstance(inp_node, TensorNode):
                for inp_node_parent in inp_node.parents:
                    if isinstance(inp_node_parent, FunctionNode):
                        input_ids.append(inp_node_parent.id)
        for out_node in self.children:
            if isinstance(out_node, TensorNode):
                for out_node_child in out_node.children:
                    if isinstance(out_node_child, FunctionNode):
                        output_ids.append(out_node_child.id)

        # iterate through module named parameters and create AbstractNNLayerParam objects
        param_list = []
        # TODO: Add a filter?
        all_properties = {
            attr: getattr(self.module_info, attr) \
            for attr in dir(self.module_info) \
            if not callable(getattr(self.module_info, attr)) \
            and not attr.startswith('_') and not attr in SKIP_LAYER_PARAM
        }

        layer_weight = None
        for name, param in all_properties.items():
            if name == 'weight':
                layer_weight = param
            else:
                named_param = AbstractNNLayerParam(
                    param_name=name,
                    param_value=param,
                )
                param_list.append(named_param)

        # modify the name of the operation if the function is contained in a module
        if self.contained_in_module:
            assert self.module_info is not None, "Module information is missing."
            operation_name = str(self.module_info)[:str(self.module_info).find('(')]
        else:
            operation_name = str(self.operation)

        # create an AbstractNNLayer object
        ann_layer = AbstractNNLayer(
            node_id=self.id,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            input_shape=[inp.shape for inp in input_tensors],
            output_shape=[out.shape for out in output_tensors],
            operation=operation_name,
            parameters=param_list,
            is_input_node=bool(not self.parents),
            is_output_node=bool(not self.children),
            weight=layer_weight if get_weight else None,
        )

        return ann_layer, (input_ids, output_ids)

    def __repr__(self):
        return f'<FunctionNode [{self.id}] operation={self.operation} module={self.module_info}> -> {[child.id for child in self.children]}'