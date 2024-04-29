"""
This module contains the Tracer class which is used to 
trace the computation graph of a PyTorch model.
"""
import torch
import torchvision.models as models
from transformers import AutoModel, AutoProcessor
from node import Node, TensorNode, FunctionNode
from ANN.abstract_neural_network import AbstractNN
from ANN.ann_layer import AbstractNNLayer

class TracerTensor(torch.Tensor):
    """
    A wrapper around torch.Tensor that adds tracing functionality to tensors.
    """
    # ID counter for tensors
    _next_id = 0

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        instance = torch.Tensor._make_subclass(cls, x, *args, **kwargs)
        instance.var_init()
        return instance

    def var_init(self):
        """
        This method initializes the attributes of the TracerTensor object.
        __init__ cannot be used because it is not called when a tensor is created.

        Attributes:
            id: A unique identifier for the tensor.
            node: A TensorNode object that represents the tensor.
            creation_op: The operation that created the tensor.
            inputs: A list of input tensors to the operation that created the tensor.
        """
        self.assign_id()
        if not hasattr(self, 'node'):
            self.node = TensorNode(self)
        if not hasattr(self, 'creation_op'):
            self.creation_op = None
        if not hasattr(self, 'inputs'):
            self.inputs = []

    def assign_id(self):
        """
        Assigns a unique identifier to the tensor.
        """
        if not hasattr(self, 'id'):
            self.id = TracerTensor._next_id
            TracerTensor._next_id += 1

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        excluded_functions = {"__get__", "__getitem__", "__repr__", "__dir__", "dim"}

        # Avoid processing non-tensor operations
        if func.__name__ in excluded_functions:
            return super().__torch_function__(func, types, args, kwargs)
        
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, torch.Tensor):
            ret = TracerTensor(ret) if not isinstance(ret, TracerTensor) else ret
            func_node = FunctionNode(func.__name__)
            input_nodes = [arg.node for arg in args if isinstance(arg, TracerTensor)]
            for node_ in input_nodes:
                node_.add_child(func_node)
            if isinstance(ret, TracerTensor):
                ret.node = TensorNode(ret)
                func_node.add_child(ret.node)
        elif not isinstance(ret, torch.Tensor):
            # Log non-tensor returns for debug purposes
            print(f"Function {func.__name__} returned a non-tensor of type {type(ret)}.")
        return ret

class TracerModule():
    """
    This class wraps a PyTorch module and replaces its 
    forward method with a wrapped version.

    Attributes:
        module: The PyTorch module to be wrapped.
    """
    def __init__(self, module):
        self.module = module
        self.visited = False
        self.io_pairs = []
        self.wrap_forward()

    def wrap_forward(self):
        """
        This method wraps the forward method of the module.
        """
        original_forward = self.module.forward

        def wrapped_forward(*inputs, **kwargs):
            self.visited = True
            new_inputs = [
                TracerTensor(inp) \
                if not isinstance(inp, TracerTensor) \
                else inp for inp in inputs
            ]
            output = original_forward(*new_inputs, **kwargs)
            if not isinstance(output, TracerTensor):
                output = TracerTensor(output)
            output.var_init()

            # Store the input-output pair in case the
            # module is called multiple times
            self.io_pairs.append((new_inputs, output))

            for inp in new_inputs:
                for child in inp.node.children:
                    # it is garanteed to be a FunctionNode
                    child.contained_in_module = True
                    child.module_info = self.module # TODO: fix bug here
            return output

        self.module.forward = wrapped_forward

class Tracer:
    """
    This class wraps a PyTorch model and replaces the forward method of its
    modules with a wrapped version that adds tracing functionality.
    """
    def __init__(self, model):
        self.model = model
        self.tracer_module_list = []
        self.input_tensor = None
        self.wrap_modules(model)

    def wrap_modules(self, module):
        """
        This method wraps the forward method of the module and its children.

        Args:
            module: The PyTorch module to be wrapped.
        """
        if not list(module.children()):
            tracer_module = TracerModule(module)
            self.tracer_module_list.append(tracer_module)
        else:
            for child in module.children():
                self.wrap_modules(child)

    def trace(self, input_tensor):
        """
        This method traces the computation graph of the model.

        Args:
            input_tensor: The input tensor to the model.
        
        Returns:
            output: The output tensor of the model.
        """
        if not isinstance(input_tensor, TracerTensor):
            input_tensor = TracerTensor(input_tensor)
        self.input_tensor = input_tensor
        output = self.model(input_tensor)
        return output
    
    def get_input_tensornode(self):
        """
        This method returns the input TensorNode object.

        Returns:
            input_tensor: The input TensorNode object.
        """
        assert self.input_tensor is not None, 'Input tensor is not set.'
        return self.input_tensor.node
    
    def to_ann(self):
        """
        This method converts the traced computation graph to an ANN object.

        Returns:
            ann: The ANN object representing the computation graph.
        """
        # create ann layer list and connection info
        ann_layer_list = []
        connection_info = []

        # for each module node in the graph, create an AbstractNNLayer object
        visited = set()
        def traverse(node):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, FunctionNode):
                ann_layer, io_ids = node.to_annlayer()
                ann_layer_list.append(ann_layer)
                # only include output connection
                connection_info.append((ann_layer.node_id, io_ids[1]))
            for child in node.children:
                traverse(child)
        traverse(self.get_input_tensornode())

        # print(ann_layer_list)
        # print('-=-=-=-=-=-=-=-=-')
        # print(connection_info)
        for node in sorted(list(visited), key=lambda x: x.id):
            print(node)

        # create an ANN object
        ann = AbstractNN(
            annlayer_list=ann_layer_list,
            connection_info=connection_info,
        )

        return ann

if __name__ == "__main__":
    # Example usage
    # model = models.resnet18(pretrained=True)
    # tracer = Tracer(model)
    # dummy_input = torch.randn(1, 3, 224, 224)
    # output = tracer.trace(dummy_input)

    # ann = tracer.to_ann()
    # print(ann)
    # load microsoft/resnet-50 from huggingface
    model = AutoModel.from_pretrained("microsoft/resnet-50")
    tracer = Tracer(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = tracer.trace(dummy_input)
    ann = tracer.to_ann()
    ann.export_ann('test_ann.json')