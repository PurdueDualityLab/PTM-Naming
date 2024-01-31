from ANN.AbstractNNLayerParam import AbstractNNLayerParam


import onnx
from onnx import NodeProto
from torchview.computation_node.base_node import Node
from torchview.computation_node.compute_node import FunctionNode, ModuleNode, TensorNode


from typing import List, Tuple


class AbstractNNLayer():

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
        parameters: List[AbstractNNLayerParam] = None,
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
    def from_torchview_modulenode(self, node: ModuleNode) -> None:
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name
        self.parameters = []

        for attr_name, attr_val in node.module_unit.__dict__.items():
            if attr_name[0] != '_' and attr_name not in self.UNUSED_PARAM_SET and 'All' not in self.UNUSED_PARAM_SET: # include non-private attributes
                self.parameters.append(AbstractNNLayerParam(attr_name, attr_val))

        if 'All' in self.UNUSED_PARAM_SET: self.parameters = None

    # fill in the class var for tensor node type
    def from_torchview_tensornode(self, node: TensorNode) -> None:
        self.node_id = node.node_id
        if node.name == 'auxiliary-tensor':
            self.is_input_node = True
            self.output_shape = node.tensor_shape
        if node.name == 'output-tensor':
            self.is_output_node = True
            self.input_shape = node.tensor_shape

    # fill in the class var for function node type
    def from_torchview_functionnode(self, node: FunctionNode) -> None:
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name

    def from_onnx(
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
                    self.parameters.append(AbstractNNLayerParam(attr.name, attr.f))
                elif attr.type == onnx.AttributeProto.INT:
                    self.parameters.append(AbstractNNLayerParam(attr.name, attr.i))
                elif attr.type == onnx.AttributeProto.STRING:
                    self.parameters.append(AbstractNNLayerParam(attr.name, attr.s))
                elif attr.type == onnx.AttributeProto.TENSOR:
                    self.parameters.append(AbstractNNLayerParam(attr.name, attr.t))
                elif attr.type == onnx.AttributeProto.GRAPH:
                    self.parameters.append(AbstractNNLayerParam(attr.name, attr.g))
                elif attr.type == onnx.AttributeProto.INTS:
                    self.parameters.append(AbstractNNLayerParam(attr.name, tuple(attr.ints)))
                elif attr.type == onnx.AttributeProto.FLOATS:
                    self.parameters.append(AbstractNNLayerParam(attr.name, tuple(attr.floats)))
                elif attr.type == onnx.AttributeProto.STRINGS:
                    self.parameters.append(AbstractNNLayerParam(attr.name, tuple(attr.strings)))
                elif attr.type == onnx.AttributeProto.TENSORS:
                    self.parameters.append(AbstractNNLayerParam(attr.name, tuple(attr.tensors)))
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    self.parameters.append(AbstractNNLayerParam(attr.name, tuple(attr.graphs)))
                else:
                    self.parameters.append(AbstractNNLayerParam(attr.name, None))

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
    def from_torchview(self, node: Node) -> None:
        if type(node) == ModuleNode:
            self.from_torchview_modulenode(node)
        elif type(node) == TensorNode:
            self.from_torchview_tensornode(node)
        elif type(node) == FunctionNode:
            self.from_torchview_functionnode(node)

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