
from ANN.AbstractNN import *
from transformers import AutoModel
import torch

class ANNGeneratorTest():

    def __init__(self):
        pass

    def test_HF_img_224_template(self, hf_repo_name):
        model = AutoModel.from_pretrained(hf_repo_name)
        ann = AbstractNN.from_huggingface(model, torch.randn(1, 3, 224, 224))
        vec_lc, vec_lp = ann.vectorize()
        print(ann)
        print(vec_lc)
        print(vec_lp)

    def test_HF_resnet18(self):
        self.test_HF_img_224_template("microsoft/resnet-18")

if __name__ == "__main__":
    test = ANNGeneratorTest()
    test.test_HF_resnet18()