
from typing import List
from onnx import GraphProto
from loguru import logger
import time
from transformers import AutoModel
from ANN.AbstractNNGenerator import AbstractNNGenerator
from ANN.AbstractNNLayer import AbstractNNLayer

from ANN.pipelines.ANNToJSONConverter import read_annlayer_list_from_json, annlayer_list_to_json

# high-level wrapper
class AbstractNN():
    def __init__(
        self,
        annlayer_list=None,
        connection_info=None
    ):
        self.content = annlayer_list
        self.connection_info = connection_info
        self.layer_connection_vector, self.layer_with_parameter_vector = \
            self.vectorize()

    def from_huggingface(hf_repo_name, tracing_input, verbose=True):
        
        if verbose: logger.info(f"Looking for model in {hf_repo_name}...")
        model = AutoModel.from_pretrained(hf_repo_name)
        if verbose: 
            logger.success(f"Successfully load the model.")
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
    
    # TODO: need some name changes
    def vectorize(self):
        """
        Vectorize the model using an n-gram like approach

        Unused Function
        """
        l_l, c_i = self.content, self.connection_info

        def get_freq_vec_l(layer_list, connection_info):
            freq_vec = dict()
            id_to_node_map = dict()
            for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
                id_to_node_map[layer_connection_info[0]] = layer_node

            def make_node_string(n: AbstractNNLayer):
                if n.is_input_node:
                    return '[INPUT]'
                if n.is_output_node:
                    return '[OUTPUT]'
                return '{} {}->{}'.format(n.operation, n.input_shape, n.output_shape)

            for layer_node, layer_connection_info in zip(layer_list, connection_info):
                curr_node_str = make_node_string(layer_node)
                for next_layer_id in layer_connection_info[1]:
                    next_node_str = make_node_string(id_to_node_map[next_layer_id])
                    combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                    if combined_str not in freq_vec:
                        freq_vec[combined_str] = 0
                    freq_vec[combined_str] += 1

            return freq_vec

        def get_freq_vec_p(layer_list: List[AbstractNNLayer]):
            freq_vec = dict()
            for l in layer_list:
                p_str_list = []
                if l.parameters != None:
                    for p in l.parameters:
                        p_str_list.append('<{}, {}>'.format(p.param_name, p.param_value))
                if l.is_input_node:
                    l_str = '[INPUT]'
                elif l.is_output_node:
                    l_str = '[OUTPUT]'
                else:
                    l_str = '{} {}'.format(l.operation, p_str_list if len(p_str_list) else '')
                if l_str not in freq_vec:
                    freq_vec[l_str] = 0
                freq_vec[l_str] += 1
            return freq_vec

        def get_freq_vec_pl(layer_list, connection_info):
            freq_vec = dict()
            id_to_node_map = dict()
            for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
                id_to_node_map[layer_connection_info[0]] = layer_node

            def make_node_string(n: AbstractNNLayer):
                if n.is_input_node:
                    return '[INPUT]'
                if n.is_output_node:
                    return '[OUTPUT]'
                return n.operation

            for layer_node, layer_connection_info in zip(layer_list, connection_info):
                curr_node_str = make_node_string(layer_node)
                for next_layer_id in layer_connection_info[1]:
                    next_node_str = make_node_string(id_to_node_map[next_layer_id])
                    combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                    if combined_str not in freq_vec:
                        freq_vec[combined_str] = 0
                    freq_vec[combined_str] += 1

            return freq_vec

        # def get_freq_vec_d(layer_list):
        #     freq_vec_d, freq_vec_dn = dict(), dict()
        #     for l in layer_list:
        #         d_list = []
        #         if l.input_shape != None:
        #             for s_in in l.input_shape:
        #                 d_list.append(s_in)
        #         if l.output_shape != None:
        #             for s_out in l.output_shape:
        #                 d_list.append(s_out)
        #         for t in d_list:
        #             if str(t) not in freq_vec_d:
        #                 freq_vec_d[str(t)] = 0
        #             freq_vec_d[str(t)] += 1
        #             for n in t:
        #                 if n not in freq_vec_dn:
        #                     freq_vec_dn[n] = 0
        #                 freq_vec_dn[n] += 1

            return freq_vec_d, freq_vec_dn

        fv_l = get_freq_vec_l(l_l, c_i)
        fv_p = get_freq_vec_p(l_l)
        fv_pl = get_freq_vec_pl(l_l, c_i)
        # fv_d, fv_dn = get_freq_vec_d(l_l)

        return fv_pl, fv_p

    def __repr__(self):
        str_ret = ""
        for layer in self.content:
            str_ret += str(layer)
            str_ret += '\n'
        return str_ret

if __name__ == "__main__":
    pass