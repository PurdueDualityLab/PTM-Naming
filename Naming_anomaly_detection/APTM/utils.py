"""
This file contains utility functions for the ANN module.
"""

import torchview

def overwrite_torchview_func():
    """
    This function overwrites the __init__ method of the TensorNode 
    and ModuleNode classes in the torchview package.
    """
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
