

from typing import Any, List, Tuple
import torch
import torchview
from loguru import logger
from torch import Tensor
from torchview.computation_graph import ComputationGraph
from tqdm import tqdm
from ANN.AbstractNNSorter import AbstractNNSorter
from ANN.ann_conversion_handler import AbstractNNConversionHandler
from ANN.ann_layer import AbstractNNLayer
from ANN.utils import overwrite_torchview_func


class AbstractNNGenerator():

    def __init__(
        self,
        model: Any,
        inputs: Tuple[Tensor, ...] = None,
        framework: str = None,
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

        if framework == None:
            framework = "pytorch"
            logger.warning("Framework unspecified, using 'pytorch'.")


        overwrite_torchview_func()

    def get_annlayer_list(self) -> List[AbstractNNLayer]:
        """
        Returns an ordered list for the model

        Parameters:
        None

        Returns:
        list: A list of NodeInfo objects which contains all the information
        of a layer in the model
        """
        if self.framework == 'pytorch':
            return self.generate_ann_from_pytorch_model(self.model, self.inputs, use_hash=self.use_hash)
        if self.framework == 'onnx':
            return self.generate_ann_from_onnx_model(self.model, use_hash=self.use_hash)

    def get_connection(self) -> List[Tuple[int, List[int]]]:
        """
        Return an ordered list for the model as well as the connection information,
        essentially making the ordered list an adjacency list.

        Parameters:
        None

        Returns:
        list: An ordered list with connection information
        """
        if self.framework == 'pytorch':
            l = self.generate_ann_from_pytorch_model_with_id_and_connection(self.model, self.inputs, use_hash=self.use_hash)
        if self.framework == 'onnx':
            l = self.generate_ann_from_onnx_model_with_id_and_connection(self.model, use_hash=self.use_hash)
        return l[0], l[1]

    def generate_ann_from_pytorch_model(
            model: Any,
            inputs: Tuple[torch.Tensor],
            graph_name: str = 'Untitled',
            depth: int = 16,
            use_hash: bool = False
        ) -> List[AbstractNNLayer]:



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
            model: Any,
            use_hash: bool = False
        ) -> List[AbstractNNLayer]:
        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_onnx(model)

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()
        return l

    def generate_ann_from_onnx_model_with_id_and_connection(
            model: Any,
            use_hash: bool = False
        ) -> List[Tuple[int, List[int]]]:

        mapper = AbstractNNConversionHandler()
        mapper.populate_class_var_from_onnx(model)

        #print('Mapper Populated')

        traverser = AbstractNNSorter(mapper, use_hash)
        l = traverser.generate_annlayer_list()

        #print('Ordered List Generated')

        layer_id_connection_list: List[Tuple[int, List[int]]] = []

        for layer in l:
            connection_list = traverser.adj_dict[layer.node_id] if layer.node_id in traverser.adj_dict else []
            connection_id_list = []
            for node in connection_list:
                connection_id_list.append(node.node_id)
            layer_id_connection_list.append((layer.node_id, connection_id_list))
        return l, layer_id_connection_list

    def generate_ann_from_pytorch_model_with_id_and_connection(
            model: Any,
            inputs: Tuple[torch.Tensor],
            graph_name: str = 'Untitled',
            depth: int = 16,
            use_hash: bool = False
        ) -> List[Tuple[int, List[int]]]:

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

        layer_id_connection_list: List[Tuple[int, List[int]]] = []

        for layer in l:
            connection_list = traverser.adj_dict[layer.node_id] if layer.node_id in traverser.adj_dict else []
            connection_id_list = []
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
        # check integrity
        if self.framework == "pytorch" and self.inputs == None:
            raise ValueError("PyTorch framework need an input to trace the computation graph.")
        elif self.framework not in ["pytorch", "onnx"]:
            raise ValueError(f"Unsupported framework {self.framework}.")

        iter_bar = tqdm(range(2 + int(self.framework=='pytorch')))

        conversion_handler = AbstractNNConversionHandler()

        if self.framework == 'pytorch':

            if self.verbose:
                iter_bar.set_description(f"Tracing PyTorch Model")
                iter_bar.update(1)

            model_graph: ComputationGraph = torchview.draw_graph(
                self.model, self.inputs,
                graph_name=graph_name,
                depth=depth,
                expand_nested=True
            )

            if self.verbose:
                iter_bar.set_description(f"Converting torch Graph to ANN")
                iter_bar.update(1)

            conversion_handler.populate_class_var_from_torchview(model_graph.edge_list)

        elif self.framework == 'onnx':

            if self.verbose:
                iter_bar.set_description(f"Converting onnx Graph to ANN")
                iter_bar.update(1)

            conversion_handler.populate_class_var_from_onnx(self.model)

        if self.verbose:
            iter_bar.set_description(f"Converting onnx Graph to ANN")
            iter_bar.update(1)

        traverser = AbstractNNSorter(conversion_handler, self.use_hash)
        annlayer_list = traverser.generate_annlayer_list()

        if include_connection:
            layer_id_connection_list: List[Tuple[int, List[int]]] = []
            for layer in annlayer_list:
                connection_list = traverser.adj_dict[layer.node_id] if layer.node_id in traverser.adj_dict else []
                connection_id_list = []
                for node in connection_list:
                    connection_id_list.append(node.node_id)
                layer_id_connection_list.append((layer.node_id, connection_id_list))
            return annlayer_list, layer_id_connection_list
        else:
            return annlayer_list