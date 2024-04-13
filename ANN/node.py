
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
    
    def __repr__(self):
        return f'<Node [{self.id}]> -> {[child.id for child in self.children]}'
    
class TensorNode(Node):
    def __init__(self, tensor):
        super().__init__()
        self.tensor = tensor
        self.shape = tensor.shape

    def __repr__(self):
        return f'<TensorNode [{self.id}] shape={self.shape}> -> {[child.id for child in self.children]}'

class FunctionNode(Node):
    def __init__(self, operation):
        super().__init__()
        self.operation = operation
        self.contained_in_module = False
        self.module_info = None

    def __repr__(self):
        return f'<FunctionNode [{self.id}] operation={self.operation} module={self.module_info}> -> {[child.id for child in self.children]}'