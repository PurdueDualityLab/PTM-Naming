"""
This class is used to represent a parameter of a neural network layer.
"""
from typing import Any

class AbstractNNLayerParam():
    """
    This class is used to represent a parameter of a neural network layer.

    Attributes:
        param_name: The name of the parameter
    """
    def __init__(
        self,
        param_name: str,
        param_value: Any = None
    ) -> None:
        self.param_name = param_name
        self.param_value = param_value

    def __eq__(self, other) -> bool:
        if self.param_name != other.param_name:
            return False
        return self.param_value == other.param_value

    def __str__(self) -> str:
        return '<' + self.param_name + ': ' + str(self.param_value) + '>'

    def __hash__(self) -> int:
        return hash(str(self.param_name) + str(self.param_value))