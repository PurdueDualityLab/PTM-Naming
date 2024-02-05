
from typing import List
from onnx import GraphProto
from loguru import logger
import time
from transformers import AutoModel
from ANN.AbstractNNGenerator import AbstractNNGenerator
from ANN.AbstractNNLayer import AbstractNNLayer
from tools.HFValidInputIterator import HFValidInputIterator

from ANN.old_pipelines.ANNToJSONConverter import read_annlayer_list_from_json, annlayer_list_to_json

# high-level wrapper
class AbstractNN():
    def __init__(
        self,
        annlayer_list=None,
        connection_info=None
    ):
        self.content = annlayer_list
        self.connection_info = connection_info
        self.layer_connection_vector, self.layer_with_parameter_vector, self.dim_vector = \
            self.vectorize()

    def from_huggingface(hf_repo_name, tracing_input="auto", verbose=True):
        
        if verbose: logger.info(f"Looking for model in {hf_repo_name}...")
        model = AutoModel.from_pretrained(hf_repo_name)
        if verbose: 
            logger.success(f"Successfully load the model.")

        if tracing_input == "auto":
            if verbose: logger.info(f"Automatically generating an input...")
            in_iter = HFValidInputIterator(model, hf_repo_name, cache_dir=None) # TODO: modify cache_dir
            tracing_input = in_iter.get_valid_input()
            if verbose: logger.success(f"Successfully generating an input.")

        if verbose:
            logger.info(f"Generating ANN...")

        start_time = time.time()

        ann_gen = AbstractNNGenerator(
            model=model,
            inputs=tracing_input,
            framework="pytorch",
            use_hash=True,
            verbose=True
        )
        
        layer_list, conn_info = ann_gen.generate_annlayer_list(include_connection=True)

        end_time = time.time()

        if verbose:
            logger.success(f"ANN generated. Time taken: {round(end_time - start_time, 4)}s")
            logger.info("Vectorizing...")

        ret_ann = AbstractNN(layer_list, conn_info)

        if verbose:
            logger.success("Success.")
        return ret_ann
    
    def from_json(json_loc):
        layer_list, conn_info = read_annlayer_list_from_json(json_loc)
        return AbstractNN(layer_list, conn_info)

    def export_json(self, output_loc):
        annlayer_list_to_json(self.content, self.connection_info, output_loc)
    
    def get_annlayer_iodim_repr(self, n: AbstractNNLayer):
        if n.is_input_node:
            return '[INPUT]'
        if n.is_output_node:
            return '[OUTPUT]'
        return '{} {}->{}'.format(n.operation, n.input_shape, n.output_shape)

    def get_layer_vector(self):
        freq_vec = dict()
        id_to_node_map = dict()
        for layer_node, layer_connection_info in zip(self.content, self.connection_info): # assume no repetitive layer in ordered list
            id_to_node_map[layer_connection_info[0]] = layer_node

        for layer_node, layer_connection_info in zip(self.content, self.connection_info):
            curr_node_str = self.get_annlayer_iodim_repr(layer_node)
            for next_layer_id in layer_connection_info[1]:
                next_node_str = self.get_annlayer_iodim_repr(id_to_node_map[next_layer_id])
                combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                if combined_str not in freq_vec:
                    freq_vec[combined_str] = 0
                freq_vec[combined_str] += 1

        return freq_vec
    
    def get_layer_param_vector(self):
        freq_vec = dict()
        for annlayer in self.content:
            layer_param_repr_list = []
            if annlayer.parameters != None:
                for param in annlayer.parameters:
                    layer_param_repr_list.append('<{}, {}>'.format(param.param_name, param.param_value))
            if annlayer.is_input_node:
                annlayer_repr = '[INPUT]'
            elif annlayer.is_output_node:
                annlayer_repr = '[OUTPUT]'
            else:
                annlayer_repr = '{} {}'.format(annlayer.operation, layer_param_repr_list if len(layer_param_repr_list) else '')
            if annlayer_repr not in freq_vec:
                freq_vec[annlayer_repr] = 0
            freq_vec[annlayer_repr] += 1
        return freq_vec
    
    def get_dim_vector(self):
        freq_vec = dict()
        for annlayer in self.content:
            dim_list = []
            if annlayer.input_shape != None:
                for dim_in in annlayer.input_shape:
                    dim_list.append(dim_in)
            if annlayer.output_shape != None:
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

        Unused Function
        """

        fv_l = self.get_layer_vector()
        fv_p = self.get_layer_param_vector()
        fv_d = self.get_dim_vector()

        return fv_l, fv_p, fv_d

    def __repr__(self):
        str_ret = ""
        for layer in self.content:
            str_ret += str(layer)
            str_ret += '\n'
        return str_ret

if __name__ == "__main__":
    pass