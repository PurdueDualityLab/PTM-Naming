
from transformers import ResNetForImageClassification
from transformers import AlbertTokenizer, AlbertModel, AutoModelForMaskedLM

from ANN.unused_code.AbstractNNComparator import ANNComparator
from ANN.abstract_neural_network import print_list
import timm
import torch

'''
comp = OrderedListComparator()
d1 = '/depot/davisjam/data/chingwo/PTM-Naming/model_for_validation/resnet101-v1-onnx.onnx' # ONNX resnet101v1
d2 = '/depot/davisjam/data/chingwo/PTM-Naming/model_for_validation/resnet101-v1-torch.onnx' # keras resnet101v1
comp.from_ONNX_model_directory(d1, d2)
comp.get_diff()
print(comp.get_ngram_cosine_similarity())
'''

comp = ANNComparator()
comp.from_NLP_model_name("albert-base-v1", "asafaya/albert-base-arabic")
print(comp.get_nlayer_cosine_similarity())
print(comp.get_param_cosine_similarity())
