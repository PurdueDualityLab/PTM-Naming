import torch
import torch.nn as nn

class TracerTensor(torch.Tensor):
    _next_id = 0  # Class variable to keep track of the next available ID

    def var_init(self):
        self.id = TracerTensor._next_id
        TracerTensor._next_id += 1
        self.creation_op = None
        self.inputs = []

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)  # Ensure dtype compatibility
        instance = torch.Tensor._make_subclass(cls, x, *args, **kwargs)
        instance.var_init()
        return instance

    def set_creation_op(self, op_description, input_tensors):
        self.creation_op = op_description
        self.inputs = [inp.id for inp in input_tensors if isinstance(inp, TracerTensor)]

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
            new_inputs = [TracerTensor(inp) if not isinstance(inp, TracerTensor) else inp for inp in inputs]
            self.input_tracer_tensor = new_inputs
            output = original_forward(*new_inputs, **kwargs)
            if not isinstance(output, TracerTensor):
                output = TracerTensor(output)
            output.var_init()
            # Record the operation and inputs in the output tensor
            output.set_creation_op(str(self.module), new_inputs)
            self.output_tracer_tensor = output
            return output

        self.module.forward = wrapped_forward

class Tracer:

    def __init__(self, model):
        self.model = model
        self.tracer_module_list = []
        self.wrap_modules(model)

    def wrap_modules(self, module):
        for child in module.children():
            tracer_module = TracerModule(child)
            self.tracer_module_list.append(tracer_module)
            tracer_module.wrap_forward()
            self.wrap_modules(child)

    def trace(self, input_tensor):
        if not isinstance(input_tensor, TracerTensor):
            input_tensor = TracerTensor(input_tensor)
        output = self.model(input_tensor)
        return output

# Example usage
model = nn.Sequential(nn.Conv2d(1, 20, 5), nn.ReLU(), nn.Conv2d(20, 1, 5))
tracer = Tracer(model)
dummy_input = torch.randn(1, 1, 28, 28)
output = tracer.trace(dummy_input)

for tracer_module in tracer.tracer_module_list:
    print(f"Module: {tracer_module.module}, Input: {[inp.id for inp in tracer_module.input_tracer_tensor]}, Output: {tracer_module.output_tracer_tensor.id}")
