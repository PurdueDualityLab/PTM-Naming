from ANN.ann_conversion_handler import AbstractNNConversionHandler
from ANN.ann_layer import AbstractNNLayer


from torchview.computation_node.compute_node import TensorNode


from typing import Dict, List, Tuple


class AbstractNNSorter():
    """
    This class is used to sort the nodes in the graph based on their sorting identifier

    Attributes:
        mapper: The mapper object that contains the graph information
        use_hash: A boolean that indicates whether to use hash or not
        input_node_info_obj_list: A list of input nodes
        output_node_info_obj_list: A list of output nodes
        adj_dict: A dictionary that contains the adjacency list of the graph
    """
    def __init__(
        self,
        mapper: AbstractNNConversionHandler,
        use_hash: bool = False
    ):
        self.mapper = mapper
        self.input_node_info_obj_list: List[TensorNode] = []
        self.output_node_info_obj_list: List[TensorNode] = []
        self.use_hash = use_hash
        for edge_node_info_tuple in mapper.ann_layer_edge_list: # identify input/output node and put them into the class var
            if edge_node_info_tuple[0].is_input_node and edge_node_info_tuple[0] not in self.input_node_info_obj_list:
                self.input_node_info_obj_list.append(edge_node_info_tuple[0])
            if edge_node_info_tuple[1].is_output_node and edge_node_info_tuple[1] not in self.output_node_info_obj_list:
                self.output_node_info_obj_list.append(edge_node_info_tuple[1])
        self.adj_dict: Dict[int, List[AbstractNNLayer]] = mapper.get_adj_dict({'remove_identity'})

    # A helper function that helps to sort a list of node based on their sorting identifiers
    def sorted_node_info_list(self, node_info_list: List[AbstractNNLayer]):
        if self.use_hash: return sorted(node_info_list, key=lambda obj: obj.sorting_hash)
        return sorted(node_info_list, key=lambda obj: obj.sorting_identifier)

    # A function that clears the visited class var for future traversals
    def reset_visited_field(self) -> None:
        node_info_obj_set = self.mapper.ann_layer_set
        for node_info_obj in node_info_obj_set:
            node_info_obj.preorder_visited = False
            node_info_obj.postorder_visited = False


    # A function that assign sorting identifier to each of the NodeInfo in postorder
    def assign_sorting_identifier(self) -> None:

        self.reset_visited_field()

        def remove_common_suffix(strlist: List[str]) -> Tuple[List[str], str]:
            # If list is empty, return it as is
            if not strlist:
                return strlist, ""

            # Initialize the common suffix to the reversed first string in the list
            suffix = strlist[0][::-1]
            min_len = len(suffix)

            # Compare the reversed strings from left to right (original strings from right to left)
            for string in strlist[1:]:
                string = string[::-1]
                min_len = min(min_len, len(string))
                for i in range(min_len):
                    if string[i] != suffix[i]:
                        min_len = i
                        break
                # Truncate the common suffix to its common part
                suffix = suffix[:min_len]

            # Remove the common suffix from each string
            if min_len > 0:
                return [string[:-min_len] for string in strlist], suffix[::-1]
            else:
                return strlist, ""

        def traverse(curr_node_info_obj: AbstractNNLayer) -> None:

            curr_node_info_obj.preorder_visited = True

            sorting_identifier, sorting_hash = None, None
            if self.use_hash: sorting_hash: int = curr_node_info_obj.generate_sorting_hash()
            else: sorting_identifier: str = curr_node_info_obj.generate_sorting_identifier_head()

            if curr_node_info_obj.node_id not in self.adj_dict: # output node

                if self.use_hash: curr_node_info_obj.sorting_hash = sorting_hash
                else: curr_node_info_obj.sorting_identifier = sorting_identifier

            else:

                for next_node_info_obj in self.adj_dict[curr_node_info_obj.node_id]:
                    if next_node_info_obj.preorder_visited or next_node_info_obj.postorder_visited: # handles cyclic graphs
                        continue
                    traverse(next_node_info_obj)

                sorted_next_obj_list = self.sorted_node_info_list(self.adj_dict[curr_node_info_obj.node_id]) # sort the next nodes


                if self.use_hash: # hash option
                    hash_sum = sorting_hash
                    for next_node_info_obj in sorted_next_obj_list:
                        hash_sum += next_node_info_obj.sorting_hash
                    curr_node_info_obj.sorting_hash = hash(hash_sum)

                else:
                    sorting_identifier_list: List[str] = []

                    for next_node_info_obj in sorted_next_obj_list:
                        sorting_identifier_list.append(next_node_info_obj.sorting_identifier)

                    sorting_identifier_list, suffix = remove_common_suffix(sorting_identifier_list)

                    for s in sorting_identifier_list:
                        sorting_identifier += s

                    sorting_identifier += suffix

                    curr_node_info_obj.sorting_identifier = sorting_identifier

            curr_node_info_obj.postorder_visited = True

            return

        for input_node_info_obj in self.input_node_info_obj_list: # Do the same traversal for all the inputs
            traverse(input_node_info_obj)



    # A function that generates a list of graph nodes based on the order of the sorting identifier
    # similar to the above func
    def generate_annlayer_list(self) -> List[AbstractNNLayer]:

        self.assign_sorting_identifier()
        self.reset_visited_field()

        sorted_inputs: List[AbstractNNLayer] = self.sorted_node_info_list(self.input_node_info_obj_list)

        ordered_layer_list: List[AbstractNNLayer] = []

        def traverse(curr_node_info_obj: AbstractNNLayer) -> None:

            curr_node_info_obj.preorder_visited = True

            ordered_layer_list.append(curr_node_info_obj)

            if curr_node_info_obj.node_id not in self.adj_dict:
                return

            next_obj_list = self.sorted_node_info_list(self.adj_dict[curr_node_info_obj.node_id])

            for next_node_info_obj in next_obj_list:

                if next_node_info_obj.preorder_visited or next_node_info_obj.postorder_visited:
                    continue

                traverse(next_node_info_obj)
                next_node_info_obj.postorder_visited = True

            return

        for input in sorted_inputs:
            traverse(input)

        return ordered_layer_list