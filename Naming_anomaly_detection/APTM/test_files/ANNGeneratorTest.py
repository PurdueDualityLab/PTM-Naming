
from APTM.abstract_neural_network import *
from transformers import AutoModel
import torch

class ANNGeneratorTest():

    def __init__(self):
        pass

    def test_HF_img_224_template(self, hf_repo_name):
        aptm = AbstractNN.from_huggingface(hf_repo_name, torch.randn(1, 3, 224, 224))
        print(aptm.layer_connection_vector)
        print(aptm.layer_with_parameter_vector)
        aptm.export_aptm("temp.json")

    def test_HF_img_224_auto_input_template(self, hf_repo_name):
        aptm = AbstractNN.from_huggingface(hf_repo_name)
        print(aptm.layer_connection_vector)
        print(aptm.layer_with_parameter_vector)
        print(aptm.dim_vector)
        aptm.export_aptm("temp.json")

    def test_HF_resnet18(self):
        self.test_HF_img_224_template("microsoft/resnet-18")

    def test_HF_resnet18_auto_input(self):
        self.test_HF_img_224_auto_input_template("microsoft/resnet-18")

if __name__ == "__main__":
    test = ANNGeneratorTest()
    #test.test_HF_resnet18()
    test.test_HF_resnet18_auto_input()