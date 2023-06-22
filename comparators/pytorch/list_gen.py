
from utils import NodeInfo, generate_ordered_layer_list_from_pytorch_model, patch, generate_ordered_layer_list_from_pytorch_model_with_id_and_connection
from typing import Any, Tuple, List
from torch import Tensor

class OrderedListGenerator():

    def __init__(
        self,
        model: Any,
        inputs: Tuple[Tensor, ...]
    ) -> None:
        self.model = model
        self.inputs = inputs
        patch()

    def get_ordered_list(self) -> List[NodeInfo]:
        return generate_ordered_layer_list_from_pytorch_model(self.model, self.inputs)


    def print_ordered_list(self) -> None:
        ordered_list = generate_ordered_layer_list_from_pytorch_model(self.model, self.inputs)
        for layer_node in ordered_list:
            print(layer_node)
    
    def print_connection(self) -> None:
        l = generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(self.model, self.inputs)
        for layer_node, connection_info in zip(l[0], l[1]):
            print('[{}] {} -> {}'.format(connection_info[0], layer_node, connection_info[1]))