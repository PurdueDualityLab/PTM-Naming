"""
This file contains the AbstractNNGenerator class which is responsible for
generating the ANNLayer list from the model. It also contains the function
to generate the connection list of the model.
"""

from typing import Any, List, Optional, Tuple, Union
import torch
import torchview
from loguru import logger
from torchview.computation_graph import ComputationGraph
from tqdm import tqdm
from ANN.ann_sorter import AbstractNNSorter
from ANN.ann_conversion_handler import AbstractNNConversionHandler
from ANN.ann_layer import AbstractNNLayer
from ANN.utils import overwrite_torchview_func


class AbstractNNGenerator():
    """
    This class is used to generate the ANNLayer list from the model

    Attributes:
        model (Any): The model object
        inputs (tuple): The input of the model for tracing
        framework (str): The framework of the model
        use_hash (bool): A boolean that indicates whether to use hash or not
        verbose (bool): A boolean that indicates whether to print the progress or not
    """
    def __init__(
        self,
        model: Any,
        inputs: Any = None,
        framework: Optional[str] = None,
        use_hash: bool = False,
        verbose: bool = True
    ) -> None:
        """
        ANNGenerator constructor function

        Parameters:
        model (Any): Any model type from pytorch or onnx
        inputs (tuple): Input of the model for tracing, not necessary for
        onnx model
        mode (str): Specify the list generator on generating list for onnx
        or pytorch model
        use_hash (bool): Use hash while assigning layer identifiers, set
        it to True for increase performance

        Returns:
        None
        """
        self.model = model
        self.inputs = inputs
        self.framework = framework
        self.use_hash = use_hash
        self.verbose = verbose

        if framework is None:
            framework = "pytorch"
            logger.warning("Framework unspecified, using 'pytorch'.")


        overwrite_torchview_func()

    def get_annlayer_list(self) -> List[AbstractNNLayer]:
        """
        Returns an ordered list for the model

        Returns:
            An ordered list of AbstractNNLayer objects
        """
        if self.framework == 'pytorch':
            if self.inputs is None:
                raise ValueError("PyTorch framework need an input to trace the computation graph.")
            return self.generate_ann_from_pytorch_model(
                self.model,
                self.inputs,
                use_hash=self.use_hash
            )
        if self.framework == 'onnx':
            return self.generate_ann_from_onnx_model(self.model, use_hash=self.use_hash)
        raise ValueError(f"Unsupported framework {self.framework}.")

    def get_connection(self) -> Tuple[
            List[AbstractNNLayer],
            List[Tuple[Union[int, str], List[Union[int, str]]]]
        ]:
        """
        Return an ordered list for the model as well as the connection information,
        essentially making the ordered list an adjacency list.

        Parameters:
        None

        Returns:
        list: An ordered list with connection information
        """
        l = None
        if self.framework == 'pytorch':
            if self.inputs is None:
                raise ValueError("PyTorch framework need an input to trace the computation graph.")
            l = self.generate_ann_from_pytorch_model_with_id_and_connection(
                self.model,
                self.inputs,
                use_hash=self.use_hash
            )
        if self.framework == 'onnx':
            l = self.generate_ann_from_onnx_model_with_id_and_connection(
                self.model,
                use_hash=self.use_hash
            )
        if l is None:
            raise ValueError(f"Unsupported framework {self.framework}.")
        return l

    def generate_ann_from_pytorch_model(
            self,
            model: Any,
            inputs: Tuple[torch.Tensor],
            graph_name: str = 'Untitled',
            depth: int = 16,
            use_hash: bool = False
        ) -> List[AbstractNNLayer]:
        """
        This function generates an ordered list of AbstractNNLayer objects from a PyTorch model

        Args:
            model: The PyTorch model
            inputs: The input of the model for tracing
            graph_name: The name of the graph
            depth: The depth of the graph
            use_hash: A boolean that indicates whether to use hash or not
        
        Returns:
            An ordered list of AbstractNNLayer objects
        """

        model_graph: ComputationGraph = torchview.draw_graph(
            model, inputs,
            graph_name=graph_name,
            depth=depth,
            expand_nested=True
        )

        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_torchview(model_graph.edge_list)

        print('Mapper Populated')

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()
        print('Ordered List Generated')
        return l

    def generate_ann_from_onnx_model(
            self,
            model: Any,
            use_hash: bool = False
        ) -> List[AbstractNNLayer]:
        """
        This function generates an ordered list of AbstractNNLayer objects from an ONNX model

        Args:
            model: The ONNX model
            use_hash: A boolean that indicates whether to use hash or not
        
        Returns:
            An ordered list of AbstractNNLayer objects
        """
        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_onnx(model)

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()
        return l

    def generate_ann_from_onnx_model_with_id_and_connection(
            self,
            model: Any,
            use_hash: bool = False
        ) -> Tuple[List[AbstractNNLayer], List[Tuple[Union[int, str], List[Union[int, str]]]]]:

        """
        This function generates an ordered list of AbstractNNLayer objects from an ONNX model

        Args:
            model: The ONNX model
            use_hash: A boolean that indicates whether to use hash or not

        Returns:    
            An ordered list of AbstractNNLayer objects
        """

        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_onnx(model)

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()

        layer_id_connection_list: List[Tuple[Union[int, str], List[Union[int, str]]]] = []

        for layer in l:
            connection_list = traverser.adj_dict[layer.node_id] \
                if layer.node_id in traverser.adj_dict else []
            connection_id_list: List[Union[int, str]] = []
            for node in connection_list:
                connection_id_list.append(node.node_id)
            layer_id_connection_list.append((layer.node_id, connection_id_list))
        return l, layer_id_connection_list

    def generate_ann_from_pytorch_model_with_id_and_connection(
            self,
            model: Any,
            inputs: Tuple[torch.Tensor],
            graph_name: str = 'Untitled',
            depth: int = 16,
            use_hash: bool = False
        ) -> Tuple[List[AbstractNNLayer], List[Tuple[Union[int, str], List[Union[int, str]]]]]:
        """
        This function generates an ordered list of AbstractNNLayer objects from a PyTorch model

        Args:
            model: The PyTorch model
            inputs: The input of the model for tracing
            graph_name: The name of the graph
            depth: The depth of the graph
            use_hash: A boolean that indicates whether to use hash or not
        
        Returns:
            An ordered list of AbstractNNLayer objects
        """

        print('Preparing tracing')

        model_graph: ComputationGraph = torchview.draw_graph(
            model, inputs,
            graph_name=graph_name,
            depth=depth,
            expand_nested=True
        )

        print('Tracing successful')

        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_torchview(model_graph.edge_list)

        print('Mapper Populated')

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()

        print('Ordered List Generated')

        layer_id_connection_list: List[Tuple[Union[int, str], List[Union[int, str]]]] = []

        for layer in l:
            connection_list = traverser.adj_dict[layer.node_id] \
                if layer.node_id in traverser.adj_dict else []
            connection_id_list: List[Union[int, str]] = []
            for node in connection_list:
                connection_id_list.append(node.node_id)
            layer_id_connection_list.append((layer.node_id, connection_id_list))
        return l, layer_id_connection_list

    # wrapper for generating ann
    def generate_annlayer_list(
        self,
        graph_name: str = "NotSpecified",
        depth: int = 16,
        include_connection: bool = False
    ):
        """
        This function generates an ordered list of AbstractNNLayer objects from a model

        Args:
            graph_name: The name of the graph
            depth: The depth of the graph
            include_connection: A boolean that indicates whether to include 
                connection information or not
        
        Returns:
            An ordered list of AbstractNNLayer objects
        """
        # check integrity
        if self.framework == "pytorch" and self.inputs is None:
            raise ValueError("PyTorch framework need an input to trace the computation graph.")
        elif self.framework not in ["pytorch", "onnx"]:
            raise ValueError(f"Unsupported framework {self.framework}.")

        iter_bar = tqdm(range(2 + int(self.framework=='pytorch')))

        conversion_handler = AbstractNNConversionHandler()

        if self.framework == 'pytorch':

            if self.verbose:
                iter_bar.set_description("Tracing PyTorch Model")
                iter_bar.update(1)

            model_graph: ComputationGraph = torchview.draw_graph(
                self.model, self.inputs,
                graph_name=graph_name,
                depth=depth,
                expand_nested=True
            )

            if self.verbose:
                iter_bar.set_description("Converting torch Graph to ANN")
                iter_bar.update(1)

            conversion_handler.populate_class_var_from_torchview(model_graph.edge_list)

        elif self.framework == 'onnx':

            if self.verbose:
                iter_bar.set_description("Converting onnx Graph to ANN")
                iter_bar.update(1)

            conversion_handler.populate_class_var_from_onnx(self.model)

        if self.verbose:
            iter_bar.set_description("Converting onnx Graph to ANN")
            iter_bar.update(1)

        traverser = AbstractNNSorter(conversion_handler, self.use_hash)
        annlayer_list = traverser.generate_annlayer_list()

        if include_connection:
            layer_id_connection_list: List[Tuple[Union[int, str], List[Union[int, str]]]] = []
            for layer in annlayer_list:
                connection_list = traverser.adj_dict[layer.node_id] \
                    if layer.node_id in traverser.adj_dict else []
                connection_id_list: List[Union[int, str]] = []
                for node in connection_list:
                    connection_id_list.append(node.node_id)
                layer_id_connection_list.append((layer.node_id, connection_id_list))
            return annlayer_list, layer_id_connection_list
        return annlayer_list
