"""
abstract_neural_network.py

This file contains the AbstractNN class, which is a high-level wrapper for the APTMLayer class. 
It is used to represent the entire neural network architecture in a more human-readable way.
It also contains methods to vectorize the architecture and to convert it to and from JSON.
"""

import os
import time
import json
from typing import List, Tuple, Union, Optional, Any

import torch
from loguru import logger
import transformers
from transformers import AutoModel, AutoConfig

from APTM.aptm_generator import AbstractNNGenerator
from APTM.aptm_layer import AbstractNNLayer
from APTM.old_pipelines.APTMToJSONConverter import read_aptmlayer_list_from_json, aptmlayer_list_to_json
from tools.HFValidInputIterator_eval import HFValidInputIterator

class AbstractNN():
    """
    AbstractNN is a high-level wrapper for the APTMLayer class. 
    It is used to represent the entire neural network architecture in a more human-readable way.

    aptmlayer_list: A list of AbstractNNLayer objects
    connection_info: A list of tuples, where each tuple contains the index of the layer and a list
    of indices of the layers it is connected to
    """
    def __init__(
        self,
        aptmlayer_list: Optional[List[AbstractNNLayer]] = None,
        connection_info: Optional[List[Tuple[Union[int, str], List[Union[int, str]]]]] = None,
    ) -> None:
        self.content = aptmlayer_list
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
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = False,
        eval_return_params: bool = False,
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
        loading_start_time = time.time()
        # try:    # this can be improved by loading AutoConfig
        #     model = AutoModel.from_pretrained(
        #         hf_repo_name,
        #         trust_remote_code=trust_remote_code,
        #         **kwargs
        #     )
        # except Exception as emsg: # pylint: disable=broad-except
        #     err_msg = str(emsg)
        # if model is None:
        #     try:
        #         model = AutoModel.from_pretrained(
        #             hf_repo_name,
        #             from_tf=True,
        #             trust_remote_code=trust_remote_code,
        #             **kwargs
        #         )
        #     except Exception as emsg: # pylint: disable=broad-except
        #         err_msg = str(emsg)
        # if model is None:
        #     raise RuntimeError(f"Failed to load model from {hf_repo_name}: {err_msg}")

        # correct_arch = model.config.architectures[0] if model.config.architectures else None
        config = AutoConfig.from_pretrained(hf_repo_name)
        correct_arch = config.architectures[0] if config.architectures else None
        if correct_arch == "LLaMAForCausalLM":
            correct_arch = "LlamaForCausalLM"
        # if correct_arch and correct_arch != type(model).__name__:
        #     if verbose:
        #         logger.info(f"Reloading model as: {correct_arch}")
        try:
            model_class = getattr(transformers, correct_arch, None)
            model = model_class.from_pretrained(
                hf_repo_name,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype="auto",
                **kwargs
            )
            if verbose:
                logger.info(f"Successfully loaded model as: {correct_arch}")
        except Exception as emsg: # pylint: disable=broad-except
            err_msg = str(emsg)
            logger.error(f"Failed to load model as: {correct_arch}, {emsg}")
        if model is None:
            try:
                model = model_class.from_pretrained(
                    hf_repo_name,
                    from_tf=True,
                    trust_remote_code=trust_remote_code,
                    device_map=device_map,
                    torch_dtype="auto"
                    **kwargs
                )
            except Exception as emsg: # pylint: disable=broad-except
                err_msg = str(emsg)
        if model is None:
            try:    # this can be improved by loading AutoConfig
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
                raise RuntimeError(f"Failed to load model from {hf_repo_name}: {err_msg}")
        if model is None:
            raise ValueError(f"Failed to load the model: {err_msg}")
        model = model.to(device)
        T_loading = time.time() - loading_start_time
        
        # count total parameters (eval)
        calculate_total_params_start_time = time.time()
        total_params = sum(p.numel() for p in model.parameters())
        T_caculate_total_params = time.time() - calculate_total_params_start_time
        if eval_return_params:
            return T_loading, total_params

        if verbose:
            logger.success(f"Successfully load the model.")
        
        # start APTM generation, vectorization
        start_time = time.time()
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
                raise ValueError("Captmot find a valid autoclass.")
            tracing_input = in_iter.get_valid_input()
            if verbose:
                if isinstance(tracing_input, tuple):
                    if tracing_input[1] == "ErrMark":
                        raise ValueError(f"Failed to generate an input.\nError Report:\n{tracing_input[0]}")
                logger.success("Successfully generating an input.")

        if verbose:
            logger.info("Generating APTM...")
        
        aptm_gen = AbstractNNGenerator(
            model = model,
            inputs = tracing_input, # type: ignore
            framework = "pytorch", 
            use_hash = True,
            verbose = True
        )
        
        layer_list, conn_info = aptm_gen.generate_aptmlayer_list(include_connection=True)
        assert isinstance(layer_list, list)
        assert isinstance(conn_info, list)

        ret_aptm = AbstractNN(layer_list, conn_info)

        end_time = time.time()
        if verbose:
            logger.success(f"APTM generated. Time taken: {round(end_time - start_time, 4)}s")
            logger.info("Vectorizing...")
        
        if verbose:
            logger.success("Success.")
        T_aptm = end_time - start_time
        T_latency = {"T_aptm": T_aptm, "T_loading": T_loading, "T_exclude": T_caculate_total_params}
        
        return ret_aptm, T_latency, total_params

    @staticmethod
    def from_json(
        json_loc: str
    ) -> 'AbstractNN':
        """
        This method generates an AbstractNN object from a JSON file.

        json_loc: The location of the JSON file
        """
        layer_list, conn_info = read_aptmlayer_list_from_json(json_loc)
        return AbstractNN(layer_list, conn_info)

    def export_aptm(
        self,
        output_loc: str,
        create_json: bool = True
    ) -> None:
        """
        This method exports the AbstractNN object to a JSON file.

        output_loc: The location of the JSON file
        """
        if self.content is None or self.connection_info is None:
            raise ValueError("The content or connection_info is None.")

        if create_json:
            if not os.path.exists(os.path.dirname(output_loc)):
                os.makedirs(os.path.dirname(output_loc))
        aptmlayer_list_to_json(self.content, self.connection_info, output_loc)

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

    def get_aptmlayer_layer_op_repr(
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
            curr_node_str = self.get_aptmlayer_layer_op_repr(layer_node)
            for next_layer_id in layer_connection_info[1]:
                next_node_str = self.get_aptmlayer_layer_op_repr(id_to_node_map[next_layer_id])
                combined_str = f"({curr_node_str}, {next_node_str})"
                if combined_str not in freq_vec:
                    freq_vec[combined_str] = 0
                freq_vec[combined_str] += 1

        return freq_vec
    
    def get_sequences(self, start_node_id, n, current_sequence, sequences, id_to_node_map):
        """
        This method recursively builds sequences of up to length n starting from a given node.
        """
        assert self.connection_info is not None

        # Find connection info for the given start_node_id
        connection_info_for_node = next((conn_info for conn_info in self.connection_info if conn_info[0] == start_node_id), None)
        if connection_info_for_node is None:
            # If no connection info is found for this node, stop recursion
            return

        # If the current sequence has reached the desired length, add it to sequences and return
        if len(current_sequence) == n:
            sequences.append(current_sequence)
            return

        # Iterate over connected node IDs from the found connection info
        for next_node_id in connection_info_for_node[1]:
            # Ensure next_node_id is a string if your id_to_node_map uses string keys
            next_node_str_id = str(next_node_id)
            if next_node_str_id in id_to_node_map:
                next_node = id_to_node_map[next_node_str_id]
                new_sequence = current_sequence + [self.get_aptmlayer_layer_op_repr(next_node)]
                # Continue building sequences from the next node
                self.get_sequences(next_node_str_id, n, new_sequence, sequences, id_to_node_map)


    def get_layer_vector_ngram(self, n):
        """
        Returns a vector representation of sequences of up to n layers in the model.
        """
        if self.content is None or self.connection_info is None:
            raise ValueError("The content or connection_info is None.")
        
        id_to_node_map = {layer_connection_info[0]: layer_node for layer_node, layer_connection_info in zip(self.content, self.connection_info)}
        freq_vec = {}

        # For each node, find all sequences of up to length n
        for start_node_id in id_to_node_map:
            sequences = []
            self.get_sequences(start_node_id, n, [], sequences, id_to_node_map)
            
            # Update frequency vector with found sequences
            for sequence in sequences:
                sequence_str = ' -> '.join(sequence)  # Adjust formatting as needed
                freq_vec[sequence_str] = freq_vec.get(sequence_str, 0) + 1

        return freq_vec


    def get_layer_param_vector(self):
        """
        This method returns a vector representation of the parameters in the model.
        """
        freq_vec = {}

        if self.content is None:
            raise ValueError("The content is None.")

        for aptmlayer in self.content:
            layer_param_repr_list = []
            if aptmlayer.parameters is not None:
                for param in aptmlayer.parameters:
                    layer_param_repr_list.append(f"<{param.param_name}, {param.param_value}>")
            if aptmlayer.is_input_node:
                aptmlayer_repr = '[INPUT]'
            elif aptmlayer.is_output_node:
                aptmlayer_repr = '[OUTPUT]'
            else:
                if len(layer_param_repr_list) != 0:
                    aptmlayer_repr = f'{aptmlayer.operation} {layer_param_repr_list}'
                else:
                    # the extra space is to be compatible with the version in the database
                    aptmlayer_repr = f'{aptmlayer.operation} '
            if aptmlayer_repr not in freq_vec:
                freq_vec[aptmlayer_repr] = 0
            freq_vec[aptmlayer_repr] += 1
        return freq_vec

    def get_dim_vector(self):
        """
        This method returns a vector representation of the dimensions in the model.
        """
        freq_vec = {}

        if self.content is None:
            raise ValueError("The content is None.")

        for aptmlayer in self.content:
            dim_list = []
            if aptmlayer.input_shape is not None:
                for dim_in in aptmlayer.input_shape:
                    dim_list.append(dim_in)
            if aptmlayer.output_shape is not None:
                for dim_out in aptmlayer.output_shape:
                    dim_list.append(dim_out)
            for dim in dim_list:
                if str(dim) not in freq_vec:
                    freq_vec[str(dim)] = 0
                freq_vec[str(dim)] += 1

        return freq_vec

    def vectorize(self, ngram: int = 2):
        """
        Vectorize the model using an n-gram like approach
        """

        # dictionary key orders are not deterministic
        if ngram == 2:
            fv_l = self.get_layer_vector()
        else:
            fv_l = self.get_layer_vector_ngram(ngram)
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
    import sys
    sys.setrecursionlimit(2000) 
    # repo_name = 'kiddothe2b/biomedical-longformer-large'
    # repo_name = 'kazzand/ru-longformer-tiny-16384'
    repo_name = 'NHNDQ/nllb-finetuned-ko2en'
    PEATMOSS_PATH = './eval_peatmoss_data_path_arch'
    json_file_loc_vec = os.path.join(PEATMOSS_PATH, "vector")
    os.makedirs(json_file_loc_vec, exist_ok=True)
    output_path = os.path.join(json_file_loc_vec, f"{repo_name}.json")
    
    first_letter = repo_name[0].upper()
    model_path = os.path.join(os.getenv("LOCAL_WEIGHT_PATH", ""), first_letter, repo_name)
    logger.info(f"Model path: {model_path}")
    if os.path.exists(model_path):
        try:
            aptm, aptm_time, total_params = AbstractNN.from_huggingface(model_path, cache_dir=os.getenv("HF_HOME"))
            # aptm = AbstractNN.from_huggingface(model_path)
            # q_config = BitsAndBytesConfig(load_in_4bit=True)
        except Exception as emsg: # pylint: disable=broad-except
            logger.error(f"Error loading {repo_name}, {emsg}")
    else:
        aptm, aptm_time, total_params = AbstractNN.from_huggingface(repo_name, cache_dir=os.getenv("HF_HOME"))
        # try:
        # #     # aptm = AbstractNN.from_huggingface(repo_name, cache_dir=os.getenv("HF_HOME"))
        # # #     # q_config = BitsAndBytesConfig(load_in_4bit=True)
        # except Exception as emsg: # pylint: disable=broad-except
        #     logger.error(f"Error loading {repo_name}, {emsg}")
