"""
This file contains the class definition for the AbstractNNLayer class.
"""
from typing import List, Tuple, Optional
import onnx
from onnx import NodeProto # type: ignore
from torchview.computation_node.base_node import Node
from torchview.computation_node.compute_node import FunctionNode, ModuleNode, TensorNode
from ANN.ann_layer_param import AbstractNNLayerParam


class AbstractNNLayer():
    """
    This class is used to represent a layer in a neural network model

    Attributes:
        node_id: The unique identifier of the node
        input_shape: The shape of the input tensor
        output_shape: The shape of the output tensor
        operation: The operation performed by the layer
        parameters: A list of parameters of the layer
        is_input_node: A boolean that indicates whether the node is an input node
        is_output_node: A boolean that indicates whether the node is an output node
        sorting_identifier: The sorting identifier of the node
        sorting_hash: The sorting hash of the node
        preorder_visited: A boolean that indicates whether the node has been visited in 
        preorder traversal
        postorder_visited: A boolean that indicates whether the node has been visited in 
        postorder traversal
    """

    UNUSED_PARAM_SET = {
        'training',
        'training_mode'
    }

    PARAMETERS_DEFAULT_ONNX = {
        'BatchNormalization': {
            'epsilon': [9.999999747378752e-06],
            'momentum': [0.8999999761581421],
            'spatial': [1]
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
        input_shape: Optional[List[Tuple[int, ...]]] = None,
        output_shape: Optional[List[Tuple[int, ...]]] = None,
        operation: str = 'Undefined',
        parameters: Optional[List[AbstractNNLayerParam]] = None,
        is_input_node: bool = False,
        is_output_node: bool = False,
        sorting_identifier: Optional[str] = None,
        sorting_hash: Optional[int] = None,
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
        """
        This function fills in the class variables of the AbstractNNLayer object with the
        information from a ModuleNode object

        Args:
            node: The ModuleNode object to get the information from
        
        Returns:
            None
        """
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name
        self.parameters = []

        for attr_name, attr_val in node.module_unit.__dict__.items(): # type: ignore
            if attr_name[0] != '_' \
            and attr_name not in self.UNUSED_PARAM_SET \
            and 'All' not in self.UNUSED_PARAM_SET: # include non-private attributes
                self.parameters.append(AbstractNNLayerParam(attr_name, attr_val))

        if 'All' in self.UNUSED_PARAM_SET:
            self.parameters = None

    # fill in the class var for tensor node type
    def from_torchview_tensornode(self, node: TensorNode) -> None:
        """
        This function fills in the class variables of the AbstractNNLayer object with the
        information from a TensorNode object

        Args:
            node: The TensorNode object to get the information from
        
        Returns:
            None
        """
        self.node_id = node.node_id
        if node.name == 'auxiliary-tensor':
            self.is_input_node = True
            self.output_shape = node.tensor_shape
        if node.name == 'output-tensor':
            self.is_output_node = True
            self.input_shape = node.tensor_shape

    # fill in the class var for function node type
    def from_torchview_functionnode(self, node: FunctionNode) -> None:
        """
        This function fills in the class variables of the AbstractNNLayer object with the
        information from a FunctionNode object

        Args:
            node: The FunctionNode object to get the information from
        
        Returns:
            None
        """
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name

    def from_onnx(
            self,
            node: NodeProto = None,
            input_: Optional[List[Tuple[int, ...]]] = None,
            output: Optional[List[Tuple[int, ...]]] = None,
            is_input: bool = False,
            is_output: bool = False,
            custom_id: int = -501
        ) -> None:
        """
        This function fills in the class variables of the AbstractNNLayer object with the
        information from a NodeProto object

        Args:
            node: The NodeProto object to get the information from
            input_: The shape of the input tensor
            output: The shape of the output tensor
            is_input: A boolean that indicates whether the node is an input node
            is_output: A boolean that indicates whether the node is an output node
            custom_id: The unique identifier of the node
        
        Returns:
            None
        """
        self.node_id = custom_id
        self.input_shape = input_
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
                        if self.parameters[-1].param_value == \
                            self.PARAMETERS_DEFAULT_ONNX[self.operation][attr.name][0]:
                            self.parameters.pop()
                if attr.name in self.UNUSED_PARAM_SET:
                    self.parameters.pop()

    def param_compare(self, other) -> bool:
        """
        This function compares the parameters of the current node with the parameters of another
        node

        Args:
            other: The other node to compare with
        
        Returns:
            A boolean that indicates whether the parameters of the current node are equivalent to
            the parameters of the other node
        """
        if self.parameters is None:
            return other.parameters is None
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
        """
        This function compares the operation of the current node with the operation of another node

        Args:
            other: The other node to compare with
        
        Returns:
            A boolean that indicates whether the operation of the current node is equivalent to the
            operation of the other node
        """
        return (
            self.operation == other.operation
        )

    def dim_eq(self, other) -> bool:
        """
        This function compares the input and output shapes of the current node with the input and
        output shapes of another node

        Args:
            other: The other node to compare with
        
        Returns:
            A boolean that indicates whether the input and output shapes of the current node are
            equivalent to the input and output shapes of the other node
        """
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation
        )

    def param_eq(self, other) -> bool:
        """
        This function compares the parameters of the current node with the parameters of another
        node

        Args:
            other: The other node to compare with
        
        Returns:
            A boolean that indicates whether the parameters of the current node are equivalent to
            the parameters of the other node
        """
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation and \
            self.param_compare(other)
        )

    # A helper function for filling class var by identifying the type of the inputted node
    def from_torchview(self, node: Node) -> None:
        """
        This function fills in the class variables of the AbstractNNLayer object with the
        information from a Node object

        Args:
            node: The Node object to get the information from
        
        Returns:
            None
        """
        if isinstance(node, ModuleNode):
            self.from_torchview_modulenode(node)
        elif isinstance(node, TensorNode):
            self.from_torchview_tensornode(node)
        elif isinstance(node, FunctionNode):
            self.from_torchview_functionnode(node)

    # Generate the 'head' of the sorting identifier(sequence) for this node
    def generate_sorting_identifier_head(self) -> str:
        """
        This function generates the head of the sorting identifier for the node

        Returns:
            The head of the sorting identifier for the node
        """
        if self.is_input_node:
            return f'[INPUT/{self.output_shape}]'
        if self.is_output_node:
            return f'[OUTPUT/{self.input_shape}]'
        if self.parameters is not None:
            pm_list = []
            for pm in self.parameters:
                pm_list.append(str(pm))
            return f'[{str(self.input_shape)}/{str(self.output_shape)}' \
                f'/{self.operation}/{str(pm_list)}]'
        return f'[{str(self.input_shape)}/{str(self.output_shape)}/{self.operation}]'

    def generate_sorting_hash(self) -> int:
        """
        This function generates the sorting hash for the node

        Returns:
            The sorting hash for the node
        """
        return hash(self.generate_sorting_identifier_head())

    # to string function
    def __str__(self) -> str:
        if self.is_input_node:
            return f'[INPUT] out: {self.output_shape}'
        if self.is_output_node:
            return f'[OUTPUT] in: {self.input_shape}'
        if self.parameters is not None:
            pm_list = []
            for pm in self.parameters:
                pm_list.append(str(pm))
            return f'[{self.operation}] in: {str(self.input_shape)}' \
                f'out: {str(self.output_shape)} {str(pm_list)}'
        return f'[{self.operation}] in: {str(self.input_shape)} out: {str(self.output_shape)}'
