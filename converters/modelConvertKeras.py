# module use /depot/davisjam/data/chingwo/converter_env/modules
# module load conda-env/converter_env-py3.8.5
# onnx 1.12
# tensorflow 2.11
# protobuf 3.19
# tf-models-official 2.11
# https://github.com/onnx/tensorflow-onnx/issues/1078

import tensorflow as tf
import tf2onnx
import onnx
from collections import OrderedDict
from tf2onnx import optimizer

def keras_to_onnx(keras_model, directory):
    onnx_model, _ = tf2onnx.convert.from_keras(
        opset=16, 
        model=keras_model, 
        output_path=directory,
        optimizers = OrderedDict([
        ("optimize_transpose", optimizer.TransposeOptimizer),
        #("remove_redundant_upsample", optimizer.UpsampleOptimizer),
        ("fold_constants", optimizer.ConstFoldOptimizer),
        ("const_dequantize_optimizer", optimizer.ConstDequantizeOptimizer),
        ("loop_optimizer", optimizer.LoopOptimizer),
        # merge_duplication should be used after optimize_transpose
        # for optimize_transpose may have some trans nodes that can be merge
        ("merge_duplication", optimizer.MergeDuplicatedNodesOptimizer),
        #("reshape_optimizer", optimizer.ReshapeOptimizer),
        ("global_pool_optimizer", optimizer.GlobalPoolOptimizer),
        #("q_dq_optimizer", optimizer.QDQOptimizer),
        ("remove_identity", optimizer.IdentityOptimizer),
        #("remove_back_to_back", optimizer.BackToBackOptimizer),
        ("einsum_optimizer", optimizer.EinsumOptimizer),
        ])
    )
    return onnx_model