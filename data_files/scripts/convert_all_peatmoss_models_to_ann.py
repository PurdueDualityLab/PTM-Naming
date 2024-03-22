"""
This script is used to convert all Peatmoss models to ANN.
"""

import os
import time
import subprocess
from loguru import logger
from dotenv import load_dotenv
import torchview

import torch
from torch import nn
from torch.jit import ScriptModule

from torchview.computation_node import NodeContainer
from torchview.computation_graph import ComputationGraph
from torchview.computation_node import TensorNode
from torchview.recorder_tensor import (
    module_forward_wrapper, _orig_module_forward, RecorderTensor,
    reduce_data_info, collect_tensor_node, Recorder
)
from typing import (
    Sequence, Any, Mapping, Union, Callable, Iterable, Optional,
    Iterator, List
)

CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]

def run_process(run_count):
    """Run the main application process."""

    load_dotenv(".env")
    module_use_path = str(os.getenv("MODULE_USE_PATH"))
    module_load_path = str(os.getenv("MODULE_LOAD_PATH"))
    python_path = str(os.getenv("PYTHONPATH"))
    if not os.path.exists(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann"):
        os.makedirs(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann", exist_ok=True)
    json_file_loc = f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann"

    command = f"""
    module use {module_use_path} && 
    module load {module_load_path} && 
    export PYTHONPATH=$PYTHONPATH:{python_path} &&
    python data_files/scripts/export_ann.py -j {json_file_loc} -c {run_count}
    """

    return subprocess.Popen(command, shell=True, executable='/bin/bash')

def monitor_process(p: subprocess.Popen) -> int:
    """Monitor the main application process."""

    while True:
        result = p.poll()
        if result is not None:  # Process has exited
            logger.info(f"Process has exited with {result}.")
            return result
        time.sleep(0.1)

def forward_prop(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    device,
    model_graph: ComputationGraph,
    mode: str,
    **kwargs: Any,
) -> None:
    '''Performs forward propagation of model on RecorderTensor
    inside context to use module_forward_wrapper'''
    saved_model_mode = model.training
    try:
        if mode == 'train':
            model.train()
        elif mode == 'eval':
            model.eval()
        else:
            raise RuntimeError(
                f"Specified model mode not recognized: {mode}"
            )
        new_module_forward = module_forward_wrapper(model_graph)
        with Recorder(_orig_module_forward, new_module_forward, model_graph):
            with torch.no_grad():
                model = model.to(device)
                if isinstance(x, (list, tuple)):
                    _ = model(*x, **kwargs)
                elif isinstance(x, Mapping):
                    _ = model(**x, **kwargs)
                else:
                    # Should not reach this point, since process_input_data ensures
                    # x is either a list, tuple, or Mapping
                    raise ValueError("Unknown input type")
    except Exception as e:
        raise RuntimeError(
            "Failed to run torchgraph see error message"
        ) from e
    finally:
        model.train(saved_model_mode)

if __name__ == "__main__":
    run_count = 9999999999999
    p = run_process(run_count)
    monitor_process(p)
