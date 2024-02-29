"""
abstract_neural_network.py

This file contains the AbstractNN class, which is a high-level wrapper for the ANNLayer class. 
It is used to represent the entire neural network architecture in a more human-readable way.
It also contains methods to vectorize the architecture and to convert it to and from JSON.
"""

import os
import time
import json
from typing import List, Tuple, Union, Optional, Any
from loguru import logger
from transformers import AutoModel
import torch
from ANN.ann_generator import AbstractNNGenerator
from ANN.ann_layer import AbstractNNLayer
from ANN.old_pipelines.ANNToJSONConverter import read_annlayer_list_from_json, annlayer_list_to_json
from tools.HFValidInputIterator import HFValidInputIterator

class AbstractNN():
    """
    AbstractNN is a high-level wrapper for the ANNLayer class. 
    It is used to represent the entire neural network architecture in a more human-readable way.

    annlayer_list: A list of AbstractNNLayer objects
    connection_info: A list of tuples, where each tuple contains the index of the layer and a list
    of indices of the layers it is connected to
    """
    def __init__(
        self,
        annlayer_list: Optional[List[AbstractNNLayer]] = None,
        connection_info: Optional[List[Tuple[Union[int, str], List[Union[int, str]]]]] = None
    ) -> None:
        self.content = annlayer_list
        self.connection_info = connection_info
        self.layer_connection_vector, self.layer_with_parameter_vector, self.dim_vector = \
            self.vectorize()

    @staticmethod
    def from_huggingface(
        hf_repo_name: str,
        tracing_input: Union[str, Any] = "auto",
        verbose: bool = True,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> 'AbstractNN':
        """
        This method generates an AbstractNN object from a Hugging Face model.

        hf_repo_name: The name of the Hugging Face model
        tracing_input: The input to use when tracing the model
        """
        device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        if verbose:
            logger.info(f"Looking for model in {hf_repo_name}...")

        # try to load the model in both PyTorch and TensorFlow
        model = None
        err_msg = ""
        try:
            model = AutoModel.from_pretrained(
                hf_repo_name,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        except Exception as emsg: # pylint: disable=broad-except
            err_msg = str(emsg)
        if model is None:
            try:
                model = AutoModel.from_pretrained(
                    hf_repo_name,
                    from_tf=True,
                    trust_remote_code=trust_remote_code,
                    **kwargs
                )
            except Exception as emsg: # pylint: disable=broad-except
                err_msg = str(emsg)
        if model is None:
            raise ValueError(f"Failed to load the model: {err_msg}")

        if verbose:
            logger.success("Successfully load the model.")

        if tracing_input == "auto":
            if verbose:
                logger.info("Automatically generating an input...")
            in_iter = HFValidInputIterator(
                model,
                hf_repo_name,
                cache_dir=cache_dir,
                device=device,
                trust_remote_code=trust_remote_code
            )
            if in_iter.err_type == "requires_remote_code":
                raise ValueError("The model requires trust_remote_code to be True.")
            elif in_iter.err_type == "no_preprocessor_config":
                raise ValueError("The model does not have a preprocessor_config.json file.")
            if in_iter.valid_autoclass_obj_list == []:
                raise ValueError("Cannot find a valid autoclass.")
            tracing_input = in_iter.get_valid_input()
            if verbose:
                if isinstance(tracing_input, tuple):
                    if tracing_input[1] == "ErrMark":
                        raise ValueError(f"Failed to generate an input.\nError Report:\n{tracing_input[0]}")
                logger.success("Successfully generating an input.")

        if verbose:
            logger.info("Generating ANN...")

        start_time = time.time()

        # assert isinstance(tracing_input, torch.Tensor)

        ann_gen = AbstractNNGenerator(
            model = model,
            inputs = tracing_input, # type: ignore
            framework = "pytorch",
            use_hash = True,
            verbose = True
        )

        layer_list, conn_info = ann_gen.generate_annlayer_list(include_connection=True)

        assert isinstance(layer_list, list)
        assert isinstance(conn_info, list)

        end_time = time.time()

        if verbose:
            logger.success(f"ANN generated. Time taken: {round(end_time - start_time, 4)}s")
            logger.info("Vectorizing...")

        ret_ann = AbstractNN(layer_list, conn_info)

        if verbose:
            logger.success("Success.")
        return ret_ann

    @staticmethod
    def from_json(
        json_loc: str
    ) -> 'AbstractNN':
        """
        This method generates an AbstractNN object from a JSON file.

        json_loc: The location of the JSON file
        """
        layer_list, conn_info = read_annlayer_list_from_json(json_loc)
        return AbstractNN(layer_list, conn_info)

    def export_ann(
        self,
        output_loc: str
    ) -> None:
        """
        This method exports the AbstractNN object to a JSON file.

        output_loc: The location of the JSON file
        """
        if self.content is None or self.connection_info is None:
            raise ValueError("The content or connection_info is None.")
        annlayer_list_to_json(self.content, self.connection_info, output_loc)

    def export_vector(
        self,
        output_loc: str,
        create_json: bool = True
    ) -> None:
        """
        This method exports the vector representation of the AbstractNN object to a JSON file.

        output_loc: The location of the JSON file
        create_json: Whether to create a JSON file when the path does not exist
        """
        if self.layer_connection_vector is None \
            or self.layer_with_parameter_vector is None \
            or self.dim_vector is None:
            raise ValueError(
                "The layer_connection_vector, layer_with_parameter_vector, or dim_vector is None."
            )
        combined_vec = {
            "l": self.layer_connection_vector,
            "p": self.layer_with_parameter_vector,
            "d": self.dim_vector
        }

        if create_json:
            if not os.path.exists(os.path.dirname(output_loc)):
                os.makedirs(os.path.dirname(output_loc))

        with open(output_loc, "w", encoding="utf-8") as f:
            json.dump(combined_vec, f, indent=4)

    def get_annlayer_layer_op_repr(
        self,
        n: AbstractNNLayer
    ) -> str:
        """
        This method returns a string representation of an AbstractNNLayer object.

        n: The AbstractNNLayer object
        """
        if n.is_input_node:
            return '[INPUT]'
        if n.is_output_node:
            return '[OUTPUT]'
        return f"{n.operation}"

    def get_layer_vector(self):
        """
        This method returns a vector representation of the layers in the model.
        """
        freq_vec = {}
        id_to_node_map = {}

        if self.content is None or self.connection_info is None:
            raise ValueError("The content or connection_info is None.")

        # assume no repetitive layer in ordered list
        for layer_node, layer_connection_info in zip(self.content, self.connection_info):
            id_to_node_map[layer_connection_info[0]] = layer_node

        for layer_node, layer_connection_info in zip(self.content, self.connection_info):
            curr_node_str = self.get_annlayer_layer_op_repr(layer_node)
            for next_layer_id in layer_connection_info[1]:
                next_node_str = self.get_annlayer_layer_op_repr(id_to_node_map[next_layer_id])
                combined_str = f"({curr_node_str}, {next_node_str})"
                if combined_str not in freq_vec:
                    freq_vec[combined_str] = 0
                freq_vec[combined_str] += 1

        return freq_vec

    def get_layer_param_vector(self):
        """
        This method returns a vector representation of the parameters in the model.
        """
        freq_vec = {}

        if self.content is None:
            raise ValueError("The content is None.")

        for annlayer in self.content:
            layer_param_repr_list = []
            if annlayer.parameters is not None:
                for param in annlayer.parameters:
                    layer_param_repr_list.append(f"<{param.param_name}, {param.param_value}>")
            if annlayer.is_input_node:
                annlayer_repr = '[INPUT]'
            elif annlayer.is_output_node:
                annlayer_repr = '[OUTPUT]'
            else:
                if len(layer_param_repr_list) != 0:
                    annlayer_repr = f'{annlayer.operation} {layer_param_repr_list}'
                else:
                    # the extra space is to be compatible with the version in the database
                    annlayer_repr = f'{annlayer.operation} '
            if annlayer_repr not in freq_vec:
                freq_vec[annlayer_repr] = 0
            freq_vec[annlayer_repr] += 1
        return freq_vec

    def get_dim_vector(self):
        """
        This method returns a vector representation of the dimensions in the model.
        """
        freq_vec = {}

        if self.content is None:
            raise ValueError("The content is None.")

        for annlayer in self.content:
            dim_list = []
            if annlayer.input_shape is not None:
                for dim_in in annlayer.input_shape:
                    dim_list.append(dim_in)
            if annlayer.output_shape is not None:
                for dim_out in annlayer.output_shape:
                    dim_list.append(dim_out)
            for dim in dim_list:
                if str(dim) not in freq_vec:
                    freq_vec[str(dim)] = 0
                freq_vec[str(dim)] += 1

        return freq_vec

    def vectorize(self):
        """
        Vectorize the model using an n-gram like approach
        """

        # dictionary key orders are not deterministic
        fv_l = self.get_layer_vector()
        fv_p = self.get_layer_param_vector()
        fv_d = self.get_dim_vector()

        return fv_l, fv_p, fv_d

    def __repr__(self):
        str_ret = ""
        if self.content is None:
            return 'None'
        for layer in self.content:
            str_ret += str(layer)
            str_ret += '\n'
        return str_ret

if __name__ == "__main__":
    pass
