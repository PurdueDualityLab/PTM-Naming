"""
This module contains the Tracer class which is used to 
trace the computation graph of a PyTorch model.
"""
import json
from httpx import get
from sympy import N
import torch
import tqdm
import time
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

    def set_contained_in_module(self, module_info):
        """
        This method sets the contained_in_module attribute of the tensor's node to True.
        """
        self.contained_in_module = True
        self.module_info = module_info

    def is_contained_in_module(self):
        """
        This method returns the contained_in_module attribute of the tensor's node.

        Returns:
            contained_in_module: A flag indicating whether the tensor is contained in a module.
        """
        return self.contained_in_module if hasattr(self, 'contained_in_module') else False
    
    def get_module_info(self):
        """
        This method returns the module_info attribute of the tensor's node.

        Returns:
            module_info: Information about the module containing the tensor.
        """
        ret = self.module_info if hasattr(self, 'module_info') else None
        # reset module_info and contained_in_module
        self.module_info = None
        self.contained_in_module = False
        return ret

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
            func_node.contained_in_module = args[0].is_contained_in_module()
            if func_node.contained_in_module:
                func_node.module_info = args[0].get_module_info()
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
            global module_passed
            progress_bar.update(1)
            module_passed += 1
            #print(self.module) ## debug
            self.visited = True
            new_inputs = [
                TracerTensor(inp) \
                if not isinstance(inp, TracerTensor) \
                else inp for inp in inputs
            ]

            for inp in new_inputs:
                inp.set_contained_in_module(self.module)

            output = original_forward(*new_inputs, **kwargs)

            if not isinstance(output, TracerTensor):
                output = TracerTensor(output)

            output.var_init()

            # Store the input-output pair in case the
            # module is called multiple times
            self.io_pairs.append((new_inputs, output))

            return output

        self.module.forward = wrapped_forward

class Tracer:
    """
    This class wraps a PyTorch model and replaces the forward method of its
    modules with a wrapped version that adds tracing functionality.
    """
    def __init__(self, model, mode="eval", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.tracer_module_list = []
        self.input_tensor = None
        self.module_cnt = 0
        self.mode = mode
        self.device = device
        if mode == "eval":
            self.model.eval()
        elif mode == "train":
            self.model.train()
        self.wrap_modules(model)
        global progress_bar, module_passed
        module_passed = 0
        progress_bar = tqdm.tqdm(total=self.module_cnt, desc="Tracing")

    def wrap_modules(self, module):
        """
        This method wraps the forward method of the module and its children.

        Args:
            module: The PyTorch module to be wrapped.
        """
        if not list(module.children()):
            tracer_module = TracerModule(module)
            self.tracer_module_list.append(tracer_module)
            self.module_cnt += 1
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
        self.model.to(self.device)
        if not isinstance(input_tensor, TracerTensor):
            input_tensor = TracerTensor(input_tensor)
        self.input_tensor = input_tensor.to(self.device)
        output = self.model(self.input_tensor)
        print(f"Actual Module Coverage: {module_passed}/{self.module_cnt}")
        return output
    
    def get_input_tensornode(self):
        """
        This method returns the input TensorNode object.

        Returns:
            input_tensor: The input TensorNode object.
        """
        assert self.input_tensor is not None, 'Input tensor is not set.'
        return self.input_tensor.node
    
    def to_ann(self, weight_output=None):
        """
        This method converts the traced computation graph to an ANN object.

        Returns:
            ann: The ANN object representing the computation graph.
        """
        # create ann layer list and connection info
        ann_layer_list = []
        id_to_weight_dict = {}
        connection_info = []

        # for each module node in the graph, create an AbstractNNLayer object
        visited = set()
        def traverse(node):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node, FunctionNode):
                ann_layer, io_ids = node.to_annlayer(get_weight=weight_output is not None)
                id_to_weight_dict[ann_layer.node_id] = ann_layer.weight.data.tolist() if ann_layer.weight is not None else None
                if io_ids[0] == []:
                    ann_layer.is_input_node = True
                if io_ids[1] == []:
                    ann_layer.is_output_node = True
                ann_layer_list.append(ann_layer)
                # only include output connection
                connection_info.append((ann_layer.node_id, io_ids[1]))
            for child in node.children:
                traverse(child)
        traverse(self.get_input_tensornode())

        # print(ann_layer_list)
        # print('-=-=-=-=-=-=-=-=-')
        # print(connection_info)
        # for node in sorted(list(visited), key=lambda x: x.id):
        #     print(node)

        # create an ANN object
        ann = AbstractNN(
            annlayer_list=ann_layer_list,
            connection_info=connection_info,
        )

        if weight_output is not None:
            with open(weight_output, 'w') as f:
                json.dump(id_to_weight_dict, f)

        return ann

#write a pytorch model with if branching
import torch.functional as F
class BranchingModel(torch.nn.Module):
    def __init__(self):
        super(BranchingModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.conv3 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        if x.shape[1] > 10:
            x = self.pool(self.conv2(x))
        else:
            x = self.conv3(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # model = AutoModel.from_pretrained("microsoft/resnet-50")
    # tracer = Tracer(model)
    # dummy_input = torch.randn(1, 3, 224, 224)
    # output = tracer.trace(dummy_input)
    # ann = tracer.to_ann()
    # ann.export_ann('test_ann.json')
    model = BranchingModel()
    tracer = Tracer(model)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = tracer.trace(dummy_input)
    ann = tracer.to_ann(weight_output='test_weight.json')
    ann.export_ann('test_ann.json')