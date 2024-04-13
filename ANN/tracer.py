import torch
import torch.nn as nn
from node import Node, TensorNode, FunctionNode

class TracerTensor(torch.Tensor):
    _next_id = 0  # Class variable to keep track of the next available ID

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        instance = torch.Tensor._make_subclass(cls, x, *args, **kwargs)
        instance.var_init()
        return instance

    def var_init(self):
        self.assign_id()
        if not hasattr(self, 'node'):
            self.node = TensorNode(self)
        if not hasattr(self, 'creation_op'):
            self.creation_op = None
        if not hasattr(self, 'inputs'):
            self.inputs = []

    def assign_id(self):
        if not hasattr(self, 'id'):
            self.id = TracerTensor._next_id
            TracerTensor._next_id += 1

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, torch.Tensor):
            ret = TracerTensor(ret) if not isinstance(ret, TracerTensor) else ret
            func_node = FunctionNode(func.__name__)
            input_nodes = [arg.node for arg in args if isinstance(arg, TracerTensor)]
            for node in input_nodes:
                node.add_child(func_node)
            if isinstance(ret, TracerTensor):
                ret.node = TensorNode(ret)
                func_node.add_child(ret.node)
        return ret

class TracerModule():
    def __init__(self, module):
        self.module = module
        #self.module_node = ModuleNode(module)  # Represents the module
        self.wrap_forward()

    def wrap_forward(self):
        original_forward = self.module.forward
        
        def wrapped_forward(*inputs, **kwargs):
            new_inputs = [TracerTensor(inp) if not isinstance(inp, TracerTensor) else inp for inp in inputs]
            output = original_forward(*new_inputs, **kwargs)
            if not isinstance(output, TracerTensor):
                output = TracerTensor(output)
            output.var_init()
            # Link nodes
            # for inp in new_inputs:
            #     inp.node.add_child(self.module_node)
            # self.module_node.add_child(output.node)
            for inp in new_inputs:
                for child in inp.node.children:
                    child.contained_in_module = True
                    child.module_info = self.module
            return output

        self.module.forward = wrapped_forward

class Tracer:
    def __init__(self, model):
        self.model = model
        self.tracer_module_list = []
        self.input_tensor = None
        self.wrap_modules(model)

    def wrap_modules(self, module):
        if not list(module.children()):
            tracer_module = TracerModule(module)
            self.tracer_module_list.append(tracer_module)
        else:
            for child in module.children():
                self.wrap_modules(child)

    def trace(self, input_tensor):
        if not isinstance(input_tensor, TracerTensor):
            input_tensor = TracerTensor(input_tensor)
        self.input_tensor = input_tensor
        output = self.model(input_tensor)
        return output

import torchvision.models as models

model = models.resnet18(pretrained=True)  # You can use any variant of ResNet
tracer = Tracer(model)
dummy_input = torch.randn(1, 3, 224, 224)
output = tracer.trace(dummy_input)

# traverse the graph and print the nodes
visited = set()
def traverse(node):
    if node in visited:
        return
    visited.add(node)
    for child in node.children:
        traverse(child)

traverse(tracer.input_tensor.node)
visited = sorted(list(visited), key=lambda x: x.id) # Sort the nodes by ID
for node in visited:
    print(node)
