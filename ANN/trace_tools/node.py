"""
This module contains the Node class, which is used to represent the nodes in the computation graph.
"""
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
    def __init__(self, tensor):
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

    def __repr__(self):
        return f'<FunctionNode [{self.id}] operation={self.operation} module={self.module_info}> -> {[child.id for child in self.children]}'