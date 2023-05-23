from CalcLayerDiff import printLayerInfoHelper
from ModelConvert import torch_to_onnx
import tensorflow as tf
import tf2onnx
import onnx
import torch


# Models to test: TODO: TFMG
# ResNetV1-101 from ONNX/Keras vs. ResNetV1-101 Torchvision/TFMG

# convert resnet101-v1 in keras to onnx format

resnet101_v1_keras = tf.keras.applications.resnet.ResNet101(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000
)
resnet_v1_101_keras = tf2onnx.convert.from_keras(resnet101_v1_keras)
print(resnet_v1_101_keras)
#resnet_v1_101_onnx = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet101-v1-7.onnx')
#resnet_v1_101_tfmg =tf2onnx.convert.process_tf_graph()
# -> python -m tf2onnx.convert --checkpoint tensorflow-model-meta-file-path --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
#printLayerInfoHelper(resnet_v1_101_onnx)
'''
resnet_v1_101_torch_temp = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
torch_to_onnx(resnet_v1_101_torch_temp, '/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet101-v1-torch.onnx')
resnet_v1_101_torch = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet101-v1-torch.onnx')

# Model to test: TODO
# ResNetV2-50/101 from Keras vs. from ONNX

resnet50_v2_keras = tf.keras.applications.resnet_v2.ResNet50V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
resnet101_v2_keras = tf.keras.applications.resnet_v2.ResNet101V2(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)
resnet_v2_50_keras = tf2onnx.convert.from_keras(resnet50_v2_keras)
resnet_v2_101_keras = tf2onnx.convert.from_keras(resnet101_v2_keras)
resnet_v2_50_onnx = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet50-v2-7.onnx')
resnet_v2_101_onnx = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet101-v2-7.onnx')

# print layer info TODO
printLayerInfo() ###
'''