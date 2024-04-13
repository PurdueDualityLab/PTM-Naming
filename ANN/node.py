
class Node:
    _next_id = 0

    def __init__(self):
        self.id = Node._next_id
        Node._next_id += 1
        self.parents = []
        self.children = []

    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
        if self not in child.parents:
            child.parents.append(self)

class TensorNode(Node):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.shape = tensor.shape

class FunctionNode(Node):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation

class ModuleNode(Node):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.module_type = str(type(module))