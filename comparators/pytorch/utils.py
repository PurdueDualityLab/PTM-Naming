
import torchview
import torch
from typing import Tuple, Any, List, Dict, Set
from torchview.computation_node.base_node import Node
from torchview.computation_node.compute_node import ModuleNode, TensorNode
from torchview.computation_graph import ComputationGraph

class ParamInfo():
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
        return '<' + self.param_name + ', ' + str(self.param_value) + '>'

class NodeInfo():

    UNUSED_PARAM_SET = {

    }

    def __init__(
        self,
        node_id: int = -1,
        input_shape: List[Tuple[int, ...]] = None,
        output_shape: List[Tuple[int, ...]] = None,
        operation: str = 'Undefined',
        parameters: List[ParamInfo] = None,
        is_input_node: bool = False,
        is_output_node: bool = False,
        sorting_identifier: str = None,
        preorder_visited: bool = False,
        postorder_visited: bool = False # handles cyclic graphs
    ) -> None:
        self.node_id = node_id
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.operation = operation
        self.parameters = parameters
        self.is_input_node = is_input_node
        self.is_output_node = is_output_node
        self.sorting_identifier = sorting_identifier
        self.preorder_visited = preorder_visited
        self.postorder_visited = postorder_visited

    def __hash__(self):
        return hash(self.node_id)

    def fill_info_module_node(self, node: ModuleNode) -> None:
        self.node_id = node.node_id
        self.input_shape = node.input_shape
        self.output_shape = node.output_shape
        self.operation = node.name
        self.parameters = []
        
        for attr_name, attr_val in node.module_unit.__dict__.items():
            if attr_name[0] != '_': # include non-private attributes
                self.parameters.append(ParamInfo(attr_name, attr_val))

    def fill_info_tensor_node(self, node: TensorNode) -> None:
        self.node_id = node.node_id
        if node.name == 'auxiliary-tensor':
            self.is_input_node = True
        if node.name == 'output-tensor':
            self.is_output_node = True

    def param_compare(self, other) -> bool:
        for p in self.parameters:
            if p not in other.parameters and p not in self.UNUSED_PARAM_SET:
                return False
        return True
    
    def __eq__(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation and \
            self.param_compare(other)
        )
    
    def op_eq(self, other) -> bool:
        return (
            self.operation == other.operation
        )
    
    def dim_eq(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation
        )
    
    def param_eq(self, other) -> bool:
        return (
            self.input_shape == other.input_shape and \
            self.output_shape == other.output_shape and \
            self.operation == other.operation and \
            self.param_compare(self, other)
        )
    
    def fill_info(self, node: Node) -> None:
        if type(node) == ModuleNode:
            self.fill_info_module_node(node)
        elif type(node) == TensorNode:
            self.fill_info_tensor_node(node)

    def generate_sorting_identifier_head(self) -> str:
        if self.is_input_node: return '[INPUT]'
        if self.is_output_node: return '[OUTPUT]'
        pm_list = []
        for pm in self.parameters:
            pm_list.append(str(pm))
        return '[{}/{}/{}/{}]'.format(str(self.input_shape), str(self.output_shape), self.operation, str(pm_list))

    def __str__(self) -> str:
        if self.is_input_node: return '[INPUT]'
        if self.is_output_node: return '[OUTPUT]'
        pm_list = []
        for pm in self.parameters:
            pm_list.append(str(pm))
        return '[{}] in: {} out: {} {}'.format(self.operation, str(self.input_shape), str(self.output_shape), str(pm_list))

class Mapper():

    def __init__(
        self,
        node_info_obj_set: Set[NodeInfo] = set(),
        node_id_to_node_obj_mapping: Dict[int, NodeInfo] = {},
        edge_node_info_list: List[Tuple[NodeInfo, NodeInfo]] = []
    ) -> None:
        self.node_info_obj_set = node_info_obj_set
        self.node_id_to_node_obj_mapping = node_id_to_node_obj_mapping
        self.edge_node_info_list = edge_node_info_list

    def populate_class_var(
        self,
        edge_list: List[Tuple[Node, Node]]
    ) -> None:
        for edge_tuple in edge_list:
            n_info_0 = NodeInfo()
            n_info_1 = NodeInfo()
            n_info_0.fill_info(edge_tuple[0])
            n_info_1.fill_info(edge_tuple[1])

            self.node_info_obj_set.add(n_info_0)
            self.node_info_obj_set.add(n_info_1)
            self.edge_node_info_list.append((n_info_0, n_info_1))
        
        for node_info_obj in self.node_info_obj_set:
            self.node_id_to_node_obj_mapping[node_info_obj.node_id] = node_info_obj

    def get_adj_dict(self) -> Dict[int, List[NodeInfo]]:
        adj_dict: Dict[int, List[NodeInfo]] = dict()
        for node_info_tuple in self.edge_node_info_list:
            if node_info_tuple[0].node_id not in adj_dict:
                adj_dict[node_info_tuple[0].node_id] = []
            adj_dict[node_info_tuple[0].node_id].append(node_info_tuple[1])
        return adj_dict

class Traverser():

    def __init__(
        self,
        mapper: Mapper
    ):
        self.mapper = mapper
        self.input_node_info_obj_list: List[TensorNode] = []
        self.output_node_info_obj_list: List[TensorNode] = []
        for edge_node_info_tuple in mapper.edge_node_info_list:
            if edge_node_info_tuple[0].is_input_node:
                self.input_node_info_obj_list.append(edge_node_info_tuple[0])
            if edge_node_info_tuple[1].is_output_node:
                self.output_node_info_obj_list.append(edge_node_info_tuple[1])
        self.adj_dict: Dict[int, List[NodeInfo]] = mapper.get_adj_dict()

    def sorted_node_info_list(self, node_info_list: List[NodeInfo]):
        return sorted(node_info_list, key=lambda obj: obj.sorting_identifier)

    def reset_visited_field(self) -> None:
        node_info_obj_set = self.mapper.node_info_obj_set
        for node_info_obj in node_info_obj_set:
            node_info_obj.preorder_visited = False
            node_info_obj.postorder_visited = False

    def assign_sorting_identifier(self) -> None:

        self.reset_visited_field()

        def traverse(curr_node_info_obj: NodeInfo) -> None:

            curr_node_info_obj.preorder_visited = True
            sorting_identifier: str = curr_node_info_obj.generate_sorting_identifier_head()

            #print(str(curr_node_info_obj))

            if curr_node_info_obj.node_id not in self.adj_dict: # output node
                
                curr_node_info_obj.sorting_identifier = sorting_identifier

            else:

                for next_node_info_obj in self.adj_dict[curr_node_info_obj.node_id]:
                    if next_node_info_obj.preorder_visited or next_node_info_obj.postorder_visited:
                        continue
                    traverse(next_node_info_obj)
                    next_node_info_obj.postorder_visited = True
                
                #for obj in self.adj_dict[curr_node_info_obj.node_id]: print(obj.sorting_identifier)

                sorted_next_obj_list = self.sorted_node_info_list(self.adj_dict[curr_node_info_obj.node_id])

                for next_node_info_obj in sorted_next_obj_list:
                    sorting_identifier += next_node_info_obj.sorting_identifier

                curr_node_info_obj.sorting_identifier = sorting_identifier

            return

        for input_node_info_obj in self.input_node_info_obj_list:
            traverse(input_node_info_obj)

    def generate_ordered_layer_list(self) -> List[NodeInfo]:

        self.assign_sorting_identifier()
        self.reset_visited_field()

        sorted_inputs: List[NodeInfo] = self.sorted_node_info_list(self.input_node_info_obj_list)

        ordered_layer_list: List[NodeInfo] = []

        def traverse(curr_node_info_obj: NodeInfo) -> None:

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

def generate_ordered_layer_list_from_pytorch_model(
        model: Any, 
        inputs: Tuple[torch.Tensor], 
        graph_name: str = 'Untitled',
        depth: int = 6
    ) -> List[NodeInfo]:
    
    model_graph: ComputationGraph = torchview.draw_graph(
        model, inputs,
        graph_name=graph_name,
        depth=depth, 
        expand_nested=True
    )
    
    mapper = Mapper()
    mapper.populate_class_var(model_graph.edge_list)

    traverser = Traverser(mapper)
    traverser.generate_ordered_layer_list()
        

def patch():
    def new_tn_init(
            self,
            tensor,
            depth,
            parents=None,
            children=None,
            name='tensor',
            context=None,
            is_aux=False,
            main_node=None,
            parent_hierarchy=None,
        ):
        
        old_tn_init(
            self, tensor, depth, parents, children, name, context,
            is_aux, main_node, parent_hierarchy
        )
        
        self.tensor = tensor

    old_tn_init = torchview.computation_node.TensorNode.__init__

    torchview.computation_node.TensorNode.__init__ = new_tn_init

    def new_mn_init(
            self,
            module_unit,
            depth,
            parents = None,
            children = None,
            name = 'module-node',
            output_nodes = None,
        ):
        old_mn_init(self, module_unit, depth, parents, children, name, output_nodes)
        self.module_unit = module_unit

    old_mn_init = torchview.computation_node.ModuleNode.__init__

    torchview.computation_node.ModuleNode.__init__ = new_mn_init
