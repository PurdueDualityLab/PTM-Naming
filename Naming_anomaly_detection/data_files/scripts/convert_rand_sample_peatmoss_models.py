"""
This script is used to convert the random sample of peatmoss models to a vectorized format.
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
#def run_process(idx):
def run_process(idx):
    """Run the main application process."""

    load_dotenv(".env")
    module_use_path = str(os.getenv("MODULE_USE_PATH"))
    module_load_path = str(os.getenv("MODULE_LOAD_PATH"))
    python_path = str(os.getenv("PYTHONPATH"))
    if not os.path.exists(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/vec"):
        os.makedirs(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/vec", exist_ok=True)
    if not os.path.exists(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann"):
        os.makedirs(f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann", exist_ok=True)
    json_file_loc_ann = f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/ann"
    json_file_loc_vec = f"{os.getenv('PEATMOSS_VEC_DATA_PATH')}/vector"
    model_list_json_loc = "data_files/json_files/selected_peatmoss_repos.json"

    if not os.path.exists(model_list_json_loc):
        raise FileNotFoundError(f"File {model_list_json_loc} does not exist.")

    command = f"""
    module use {module_use_path} && 
    module load {module_load_path} && 
    export PYTHONPATH=$PYTHONPATH:{python_path} &&
    export TRANSFORMERS_CACHE=/scratch/gilbreth/kim3118/.cache/huggingface
    python data_files/scripts/export_ann.py -ja {json_file_loc_ann} -jv {json_file_loc_vec} -c {idx} -s {model_list_json_loc}
    """
    # python data_files/scripts/export_vec.py -j {json_file_loc_vec} -l {model_list_json_loc}

    return subprocess.Popen(command, shell=True, executable='/bin/bash')

def monitor_process(p: subprocess.Popen) -> int:
    """Monitor the main application process."""

    while True:
        result = p.poll()
        if result is not None:  # Process has exited
            logger.info(f"Process has exited with {result}.")
            return result
        time.sleep(0.1)
        
CORRECTED_INPUT_DATA_TYPE = Optional[Union[Iterable[Any], Mapping[Any, Any]]]

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
    idx = -1
    run_count = 9999999999999
    # while idx < 9000:
        # p = run_process(run_count)
    p = run_process(run_count)
    res = monitor_process(p)
    # with open("data_files/other_files/temp_index.txt", "r", encoding="utf-8") as f:
    #     idx = int(f.readline())
        