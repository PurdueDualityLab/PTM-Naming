import torch
import torch.nn as nn

class TracerTensor(torch.Tensor):
    _next_id = 0

    def assign_id(self):
        self.id = TracerTensor._next_id
        TracerTensor._next_id += 1

    def var_init(self):
        if not hasattr(self, 'id'):
            self.assign_id()
        if not hasattr(self, 'creation_op'):
            self.creation_op = None
        if not hasattr(self, 'inputs'):
            self.inputs = []

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        instance = torch.Tensor._make_subclass(cls, x, *args, **kwargs)
        instance.var_init()
        return instance

class TracerModule():

    def __init__(self, module):
        self.module = module
        self.input_tracer_tensor = []
        self.output_tracer_tensor = None
        self.visited = False

    def wrap_forward(self):
        original_forward = self.module.forward
        
        def wrapped_forward(*inputs, **kwargs):
            self.visited = True
            # new_inputs = [TracerTensor(inp) if not isinstance(inp, TracerTensor) else inp for inp in inputs]
            new_inputs = []
            for inp in inputs:
                if not isinstance(inp, TracerTensor):
                    inp = TracerTensor(inp)
                inp.var_init()
                new_inputs.append(inp)
            self.input_tracer_tensor = new_inputs
            new_inputs_clone = [inp.clone() for inp in new_inputs] # avoid modifying the original input
            output = original_forward(*new_inputs_clone, **kwargs)
            if not isinstance(output, TracerTensor):
                output = TracerTensor(output)
            output.var_init()
            self.output_tracer_tensor = output
            return output

        self.module.forward = wrapped_forward

class Tracer:

    def __init__(self, model):
        self.model = model
        self.tracer_module_list = []
        self.wrap_modules(model)

    def wrap_modules(self, module):
        if not list(module.children()):
            tracer_module = TracerModule(module)
            self.tracer_module_list.append(tracer_module)
            tracer_module.wrap_forward()
        else:
            for child in module.children():
                self.wrap_modules(child)

    def trace(self, input_tensor):
        if not isinstance(input_tensor, TracerTensor):
            input_tensor = TracerTensor(input_tensor)
        output = self.model(input_tensor)
        return output

import torchvision.models as models

model = models.resnet18(pretrained=True)  # You can use any variant of ResNet
tracer = Tracer(model)
dummy_input = torch.randn(1, 3, 224, 224)
output = tracer.trace(dummy_input)

for tracer_module in tracer.tracer_module_list:
    print(f"Module: {tracer_module.module}, Input: {[inp.id for inp in tracer_module.input_tracer_tensor]}, Output: {tracer_module.output_tracer_tensor.id}")
