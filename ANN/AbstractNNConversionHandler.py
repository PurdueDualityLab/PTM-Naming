from ANN.AbstractNNLayer import AbstractNNLayer


from torchview.computation_node.base_node import Node


from typing import Any, Dict, List, Set, Tuple


class AbstractNNConversionHandler():

    def __init__(
        self,
        node_info_obj_set: Set[AbstractNNLayer] = None,
        node_id_to_node_obj_mapping: Dict[int, AbstractNNLayer] = None,
        ann_layer_edge_list: List[Tuple[AbstractNNLayer, AbstractNNLayer]] = None
    ) -> None:
        self.ann_layer_set = node_info_obj_set
        self.ann_layer_id_to_ann_layer_obj_mapping = node_id_to_node_obj_mapping
        self.ann_layer_edge_list = ann_layer_edge_list

    # populate the class var
    # node_info_obj_set: A set of all NodeInfo objects
    # node_id_to_node_obj_mapping: A map of [NodeInfo.node_id -> NodeInfo]
    # edge_node_info_list: A list of all the edges represented as Tuple[NodeInfo, NodeInfo], where the first NodeInfo points to the second
    def populate_class_var_from_torchview(
        self,
        edge_list: List[Tuple[Node, Node]]
    ) -> None:
        self.ann_layer_set = set()
        self.ann_layer_id_to_ann_layer_obj_mapping = {}
        self.ann_layer_edge_list = []
        for edge_tuple in edge_list:

            if edge_tuple[0].node_id not in self.ann_layer_id_to_ann_layer_obj_mapping:
                n_info_0 = AbstractNNLayer()
                n_info_0.from_torchview(edge_tuple[0])
                self.ann_layer_set.add(n_info_0)
                self.ann_layer_id_to_ann_layer_obj_mapping[n_info_0.node_id] = n_info_0
            else:
                n_info_0 = self.ann_layer_id_to_ann_layer_obj_mapping[edge_tuple[0].node_id]


            if edge_tuple[1].node_id not in self.ann_layer_id_to_ann_layer_obj_mapping:
                n_info_1 = AbstractNNLayer()
                n_info_1.from_torchview(edge_tuple[1])
                self.ann_layer_set.add(n_info_1)
                self.ann_layer_id_to_ann_layer_obj_mapping[n_info_1.node_id] = n_info_1
            else:
                n_info_1 = self.ann_layer_id_to_ann_layer_obj_mapping[edge_tuple[1].node_id]

            self.ann_layer_edge_list.append((n_info_0, n_info_1))

    def populate_class_var_from_onnx(
        self,
        onnx_model: Any
    ):
        self.ann_layer_set = set()
        self.ann_layer_id_to_ann_layer_obj_mapping = {}
        self.ann_layer_edge_list = []

        node_list = onnx_model.graph.node

        # create input name -> index map
        in2idx_map = {}

        for i in range(len(node_list)):
            for input in node_list[i].input:
                if input not in in2idx_map:
                    in2idx_map[input] = []
                in2idx_map[input].append(i)

        # create input name -> tensor shape map, this could be buggy
        '''
        inname2shape_map: dict = {
            input_tensor.name: [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            for input_tensor in onnx_model.graph.input
        }'''

        inname2shape_map: dict = {init.name: init.dims for init in onnx_model.graph.initializer}

        # Add input and output nodes
        # Assumes all node names are unique
        initializer_names = set(init.name for init in onnx_model.graph.initializer)
        actual_inputs = [inp for inp in onnx_model.graph.input if inp.name not in initializer_names]

        input_nodes = actual_inputs
        output_nodes = onnx_model.graph.output

        input_node_info_list = []
        output_node_info_list = []
        io_id_cnt = -500 # use to calculate an id for an input/output NodeInfo obj
        # input node id = -500 + node index in input_nodes
        # output node id = -500 + len(input_nodes) + node index in output_nodes
        for input in input_nodes:
            input_node_info = AbstractNNLayer()
            input_node_info.from_onnx(custom_id=io_id_cnt, is_input=True)
            self.ann_layer_set.add(input_node_info)
            self.ann_layer_id_to_ann_layer_obj_mapping[io_id_cnt] = input_node_info
            io_id_cnt += 1
            input_node_info_list.append(input_node_info)
        for output in output_nodes:
            output_node_info = AbstractNNLayer()
            output_node_info.from_onnx(custom_id=io_id_cnt, is_output=True)
            self.ann_layer_set.add(output_node_info)
            self.ann_layer_id_to_ann_layer_obj_mapping[io_id_cnt] = output_node_info
            io_id_cnt += 1
            output_node_info_list.append(output_node_info)


        output_name_set = {output for node in node_list for output in node.output}
        input_nodes_name = [i.name for i in input_nodes]
        output_nodes_name = [o.name for o in output_nodes]

        # create index -> input tensor shape map
        idx2shape_map = {}

        for i in range(len(node_list)):
            shape_list = []
            for input in node_list[i].input:
                if input in inname2shape_map and input not in input_nodes_name:
                    shape_list.append(tuple(inname2shape_map[input]))

            idx2shape_map[i] = shape_list

        #for k, v in inname2shape_map.items():
        #    print(k, v)

        for input_name, indexes in in2idx_map.items():
            # omit unused inputs
            if input_name not in output_name_set and input_name not in input_nodes_name:
                continue
            for in_idx in indexes:
                # 
                for output_name in node_list[in_idx].output:
                    # successfully find a connection
                    if output_name in output_nodes_name or output_name in in2idx_map:
                        if output_name in in2idx_map:
                            out_indexes = in2idx_map[output_name]
                        else:
                            out_indexes = [-500 + len(input_nodes) + output_nodes_name.index(output_name)]

                        if input_name in input_nodes_name:
                            start_node_info = input_node_info_list[input_nodes_name.index(input_name)]

                            # Fix issue that this program omits the first layer
                            if in_idx in self.ann_layer_id_to_ann_layer_obj_mapping:
                                end_node_info = self.ann_layer_id_to_ann_layer_obj_mapping[in_idx]
                            else:
                                end_node_info = AbstractNNLayer()
                                end_node_info.from_onnx(
                                    node = node_list[in_idx],
                                    input = idx2shape_map[in_idx],
                                    custom_id = in_idx
                                )
                                self.ann_layer_id_to_ann_layer_obj_mapping[in_idx] = end_node_info
                                self.ann_layer_set.add(end_node_info)

                            self.ann_layer_edge_list.append((start_node_info, end_node_info))
                            start_node_info = end_node_info #???
                            #

                        elif in_idx in self.ann_layer_id_to_ann_layer_obj_mapping:
                            start_node_info = self.ann_layer_id_to_ann_layer_obj_mapping[in_idx]
                        else:
                            start_node_info = AbstractNNLayer()
                            start_node_info.from_onnx(
                                node = node_list[in_idx],
                                input = idx2shape_map[in_idx],
                                custom_id = in_idx
                            )
                            self.ann_layer_id_to_ann_layer_obj_mapping[in_idx] = start_node_info
                            self.ann_layer_set.add(start_node_info)

                        for out_idx in out_indexes:

                            if out_idx in self.ann_layer_id_to_ann_layer_obj_mapping:
                                end_node_info = self.ann_layer_id_to_ann_layer_obj_mapping[out_idx]
                            else:
                                end_node_info = AbstractNNLayer()
                                end_node_info.from_onnx(
                                    node = node_list[out_idx],
                                    input = idx2shape_map[out_idx],
                                    custom_id = out_idx
                                )
                                self.ann_layer_id_to_ann_layer_obj_mapping[out_idx] = end_node_info
                                self.ann_layer_set.add(end_node_info)

                            self.ann_layer_edge_list.append((start_node_info, end_node_info))


    # returns an adjacency 'dictionary' that maps NodeInfo.node_id to a list of all the 'next node's it points to
    def get_adj_dict(self, options: Set = None) -> Dict[int, List[AbstractNNLayer]]:
        adj_dict: Dict[int, List[AbstractNNLayer]] = dict()
        for node_info_tuple in self.ann_layer_edge_list:
            if node_info_tuple[0].node_id not in adj_dict:
                adj_dict[node_info_tuple[0].node_id] = []
            adj_dict[node_info_tuple[0].node_id].append(node_info_tuple[1])

        if options != None and 'remove_identity' in options:
            for n_id, next_nodes in adj_dict.items():
                cleared = False
                while not cleared:
                    cleared = True
                    for next_node in next_nodes:
                        if next_node.operation == 'Identity':
                            cleared = False
                            adj_dict[n_id].remove(next_node)
                            for next_next_node in adj_dict[next_node.node_id]:
                                adj_dict[n_id].append(next_next_node)
                            adj_dict[next_node.node_id] = []


        return adj_dict