# module use /depot/davisjam/data/chingwo/converter_env/modules
# module load conda-env/converter_env-py3.8.5
# onnx 1.12
# tensorflow 2.11
# protobuf 3.19
# tf-models-official 2.11
# module load cuda/11.7.0
# module load gcc/6.3.0
# https://github.com/onnx/tensorflow-onnx/issues/1078

from official.vision.modeling.backbones import resnet
import tf2onnx
from collections import OrderedDict
from tf2onnx import optimizer
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import tf_slim as slim
from tf_slim.nets import resnet_v1
from tensorflow.python.framework import graph_io
from tf2onnx import tf_loader
from tf2onnx.convert import _convert_common

# Example net, end_points

'''
input_tensor = tf.placeholder(tf.float32, shape=(None,224,224,3), name='input_tensor')
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_101(input_tensor, num_classes=1000, is_training=False)
'''

def createSavedModel(
        net, 
        directory = '/depot/davisjam/data/chingwo/PTM-Naming',
        saved_model_name = 'inference_graph.pb',
    ):

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Define the model
        model = tf.identity(net, name='output_tensor')

        # Export the model to a SavedModel
        frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output_tensor"])
        graph_io.write_graph(frozen, directory, saved_model_name, as_text=False)

    with open(directory + '/' + saved_model_name, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    return graph_def

def from_graph_def_rewrite(graph_def, name=None, input_names=None, output_names=None, opset=None, custom_ops=None,
                   custom_op_handlers=None, optimizers=None, custom_rewriter=None, inputs_as_nchw=None, outputs_as_nchw=None,
                   extra_opset=None, shape_override=None, target=None, large_model=False,
                   tensors_to_rename=None, output_path=None):
    if not input_names:
        raise ValueError("input_names needs to be provided")
    if not output_names:
        raise ValueError("output_names needs to be provided")
    if not name:
        name = "unknown"
    initialized_tables = None

    with tf.device("/cpu:0"):
        with tf.Graph().as_default() as tf_graph:
            with tf_loader.tf_session(graph=tf_graph) as sess:
                tf.import_graph_def(graph_def, name='')
                frozen_graph = tf_loader.freeze_session(sess, input_names=input_names, output_names=output_names)
                input_names = tf_loader.inputs_without_resource(sess, input_names)
                frozen_graph = tf_loader.tf_optimize(input_names, output_names, graph_def)

    model_proto, external_tensor_storage = _convert_common(
        frozen_graph,
        name=name,
        continue_on_error=True,
        target=target,
        opset=opset,
        custom_ops=custom_ops,
        custom_op_handlers=custom_op_handlers,
        optimizers=optimizers,
        custom_rewriter=custom_rewriter,
        extra_opset=extra_opset,
        shape_override=shape_override,
        input_names=input_names,
        output_names=output_names,
        inputs_as_nchw=inputs_as_nchw,
        outputs_as_nchw=outputs_as_nchw,
        large_model=large_model,
        tensors_to_rename=tensors_to_rename,
        initialized_tables=initialized_tables,
        output_path=output_path)

    return model_proto, external_tensor_storage

def TFMG_to_onnx(graph_def, directory):
    from_graph_def_rewrite(
        graph_def = graph_def,
        input_names = ["input_tensor:0"],
        output_names = ["output_tensor:0"],
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
        ]),
        output_path = directory
    )
