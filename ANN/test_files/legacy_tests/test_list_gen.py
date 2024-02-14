
import torch
import torch.nn as nn
from ANN.ann_generator import AbstractNNGenerator
import onnx
from transformers import ResNetForImageClassification, AlbertForMaskedLM
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoProcessor, 
    AutoFeatureExtractor,
    BartForConditionalGeneration,
    BartTokenizer,
    BertTokenizer,
    PreTrainedTokenizerFast
)
import time
import sys
from PIL import Image

import pandas as pd
import numpy as np

# use under general env
# module use /depot/davisjam/data/chingwo/general_env/modules
# module load conda-env/general_env-py3.8.5

cache_dir = "/scratch/gilbreth/cheung59/cache_huggingface"

class TestRNN(nn.Module):
    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.LeakyReLU(inplace=inplace)
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)

        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)

        return hx
    
class TestMLP(nn.Module):

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace),
            nn.Linear(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x
    
class OrderBTest(nn.Module):
    def __init__(self):
        super(OrderBTest, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 30)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1 + out2
    
class SkipConnectionATest(nn.Module):
    def __init__(self):
        super(SkipConnectionATest, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        self.branch2 = nn.Sequential(
            nn.Identity(10, 10)
        )
        self.branch3 = nn.Sequential(
            nn.Identity(10, 10)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return out1 + out2 + out3
    
class SkipConnectionBTest(nn.Module):
    def __init__(self):
        super(SkipConnectionBTest, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        return out1 + x

class OrderATest(nn.Module):
    def __init__(self):
        super(OrderATest, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 30)
        )
        self.branch2 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        return out1 + out2
    
class BigNet(nn.Module):
    def __init__(self):
        super(BigNet, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(1000):
            self.layers.append(nn.Linear(100, 100))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(100, 10)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Block(nn.Module):
    def __init__(self, num_layers=10):
        super(Block, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(100, 100))
            self.layers.append(nn.ReLU())

    def forward(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return x + y

class BigResNet(nn.Module):
    def __init__(self):
        super(BigResNet, self).__init__()

        self.blocks = nn.ModuleList()
        for _ in range(100): 
            self.blocks.append(Block())

        self.final_layer = nn.Linear(100, 10)  

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        return x

class HugeNet(nn.Module):
    def __init__(self):
        super(HugeNet, self).__init__()

        self.layers = nn.ModuleList()
        for _ in range(10000):
            self.layers.append(nn.Linear(100, 100))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(100, 10)) 

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ParallelNet(nn.Module):
    def __init__(self):
        super(ParallelNet, self).__init__()

        self.input_layer = nn.Linear(100, 100)
        self.parallel_layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(5000)])
        self.output_layer = nn.Linear(5000*100, 10)

    def forward(self, x):
        x = self.input_layer(x)
        outputs = [layer(x) for layer in self.parallel_layers]
        x = torch.cat(outputs, dim=-1)
        x = self.output_layer(x)
        return x

def RNN_test():
    gen = AbstractNNGenerator(TestRNN(), torch.randn(2, 3))
    gen.print_connection()

def MLP_Test():
    gen = AbstractNNGenerator(TestMLP(), torch.randn(2, 128))
    gen.print_connection()

def ResNet18_Test():
    rn18 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    gen = AbstractNNGenerator(rn18, torch.randn(1, 3, 224, 224))
    gen.print_connection()

def ResNet50_Test():
    rn50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    gen = AbstractNNGenerator(rn50, torch.randn(1, 3, 224, 224))
    gen.print_connection()

def ResNet50_Hash_Comp_Test():
    print('50 TEST')
    rn50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    st = time.time()
    genN = AbstractNNGenerator(rn50, torch.randn(1, 3, 224, 224))
    #genN.print_ordered_list()
    l1 = genN.get_annlayer_list()
    ed = time.time()
    t1 = ed - st
    st = time.time()
    genH = AbstractNNGenerator(rn50, torch.randn(1, 3, 224, 224))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    print(t1, t2)
    print(l1 == l2)

def ResNet101_Hash_Comp_Test():
    print('101 TEST')
    rn101 = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")
    st = time.time()
    genN = AbstractNNGenerator(rn101, torch.randn(1, 3, 224, 224))
    #genN.print_ordered_list()
    l1 = genN.get_annlayer_list()
    ed = time.time()
    t1 = ed - st
    st = time.time()
    genH = AbstractNNGenerator(rn101, torch.randn(1, 3, 224, 224))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    print(t1, t2)
    print(l1 == l2)

def BigNet_Hash_Comp_Test():
    print('BIGNET')
    sys.setrecursionlimit(5000)
    bn = BigNet()
    st = time.time()
    genN = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genN.print_ordered_list()
    l1 = genN.get_annlayer_list()
    ed = time.time()
    t1 = ed - st
    st = time.time()
    genH = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    print(t1, t2)
    print(l1 == l2)

def BigResNet_Hash_Comp_Test():
    print('BIGRESNET')
    sys.setrecursionlimit(5000)
    bn = BigResNet()
    st = time.time()
    genN = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genN.print_ordered_list()
    l1 = genN.get_annlayer_list()
    ed = time.time()
    t1 = ed - st
    st = time.time()
    genH = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    print(t1, t2)
    print(l1 == l2)

def ParallelNet_Hash_Comp_Test():
    print('PARALLELNET')
    sys.setrecursionlimit(5000)
    bn = ParallelNet()
    st = time.time()
    genN = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genN.print_ordered_list()
    l1 = genN.get_annlayer_list()
    ed = time.time()
    t1 = ed - st
    st = time.time()
    genH = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    print(t1, t2)
    print(l1 == l2)

def HugeNet_Hash_Comp_Test():
    print('HUGENET')
    sys.setrecursionlimit(50000)
    bn = HugeNet()
    '''
    st = time.time()
    genN = OrderedListGenerator(bn, torch.randn(100, 100))
    #genN.print_ordered_list()
    l1 = genN.get_ordered_list()
    ed = time.time()
    t1 = ed - st
    '''
    st = time.time()
    genH = AbstractNNGenerator(bn, torch.randn(100, 100))
    #genH.print_ordered_list()
    l2 = genH.get_annlayer_list()
    ed = time.time()
    t2 = ed - st
    #print(t1, t2)
    #print(l1 == l2)
    print(t2)

def Order_Test():
    orderA = OrderATest()
    orderB = OrderBTest()
    i = (torch.randn(1, 10))
    genA = AbstractNNGenerator(orderA, i)
    genA.print_ann()
    print()
    genB = AbstractNNGenerator(orderB, i) 
    genB.print_ann()

def Hash_Order_Test():
    orderA = OrderATest()
    orderB = OrderBTest()
    i = (torch.randn(1, 10))
    genA = AbstractNNGenerator(orderA, i, use_hash=True)
    genA.print_ann()
    print()
    genB = AbstractNNGenerator(orderB, i, use_hash=True) 
    genB.print_ann()

def ResNet18_Onnx_Test():
    rn18 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/models/resnet18-v1-7.onnx')
    gen = AbstractNNGenerator(model=rn18, framework='onnx')
    gen.print_ann()

def ResNet101_Onnx_Test():
    rn101 = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/test_models/resnet101-v1-torch.onnx')
    gen = AbstractNNGenerator(model=rn101, framework='onnx')
    gen.print_ann()

def SkipConnection_Test():
    skta = SkipConnectionATest()
    sktb = SkipConnectionBTest()
    i = (torch.randn(1, 10))
    gena = AbstractNNGenerator(skta, i)
    gena.print_ann()
    print('-=-=-=')
    genb = AbstractNNGenerator(sktb, i)
    genb.print_ann()

def Custom_Test():
    t = AutoTokenizer.from_pretrained("bert-base-cased")
    m = AutoModel.from_pretrained('mlcorelib/deberta-base-uncased')
    i = t("Test Input", return_tensors="pt")
    gen = AbstractNNGenerator(m, i)
    gen.print_ann()

def Custom3_Test():
    t = AutoTokenizer.from_pretrained("bert-base-cased")
    m = AutoModel.from_pretrained('bert-base-cased')
    i = t("Test Input", return_tensors="pt")
    gen = AbstractNNGenerator(m, i)
    gen.print_ann()
    print('\n\n-=-=-=-=-=-==-=-==-=-=-=-=\n\n')
    t = AutoTokenizer.from_pretrained("mlcorelib/deberta-base-uncased")
    m = AutoModel.from_pretrained('mlcorelib/deberta-base-uncased')
    i = t("Test Input", return_tensors="pt")
    gen = AbstractNNGenerator(m, i)
    gen.print_ann()

def Custom2_Test():
    t = AutoTokenizer.from_pretrained("saghar/xtremedistil-l12-h384-uncased-finetuned-wikitext103")
    m = AutoModel.from_pretrained('saghar/xtremedistil-l12-h384-uncased-finetuned-wikitext103')
    i = t("Test Input", return_tensors="pt")
    gen = AbstractNNGenerator(m, i, use_hash=True)
    gen.get_connection()

def AN_TORCHONNX_Val_Test():
    an = onnx.load('/depot/davisjam/data/chingwo/PTM-Naming/model_for_validation/alexnet-torch.onnx')
    gen = AbstractNNGenerator(model=an, framework='onnx')
    gen.print_ann()

def ResNet101_comp_Test():
    d1 = '/depot/davisjam/data/chingwo/PTM-Naming/model_for_validation/resnet101-v1-onnx.onnx' # ONNX resnet101v1
    d2 = '/depot/davisjam/data/chingwo/PTM-Naming/model_for_validation/resnet101-v1-torch.onnx' # keras resnet101v1
    rn101onnx = onnx.load(d1)
    gen = AbstractNNGenerator(model=rn101onnx, framework='onnx')
    gen.print_ann()
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
    rn101t = onnx.load(d2)
    gen = AbstractNNGenerator(model=rn101t, framework='onnx')
    gen.print_ann()

def print_dict(d):
    for k, v in d.items():
        print(k, v)

def Vector_ResNet50_Test():
    rn50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    gen = AbstractNNGenerator(rn50, torch.randn(1, 3, 224, 224))
    fv1, fv2 = gen.vectorize()
    print_dict(fv1)
    print('\n-=-=-=\n')
    print_dict(fv2)

def HF_Failed_Model_Test_Fix0(model_name):
    p = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    #inp = p(text="", return_tensors="pt")
    inp = p.encode("test", return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix1(model_name):
    p = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    #inp = p(text="", return_tensors="pt")
    inp = p.encode("test", return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix2(model_name):
    image = Image.open('/depot/davisjam/data/chingwo/PTM-Naming/comparators/pytorch/000000039769.jpg')
    p = AutoProcessor.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    #inp = p(text="", return_tensors="pt")
    inp = p(images=image, return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix3(model_name):
    p = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, from_tf=True)
    #inp = p(text="", return_tensors="pt")
    inp = p('t', return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix4(model_name):
    p = AutoProcessor.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    #inp = p(text="", return_tensors="pt")
    inp = p(pd.DataFrame(), return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix5(model_name):
    p = AutoProcessor.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    inp = p(np.random.randn(1, 16000), sampling_rate=16000, return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_Test_Fix6(model_name):
    p = AutoFeatureExtractor.from_pretrained(model_name, cache_dir = cache_dir)
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    inp = p(np.random.randn(1, 16000), sampling_rate=16000, return_tensors='pt')

    gen = AbstractNNGenerator(m, inp, use_hash=True)
    gen.get_connection()

def HF_Failed_Model_0_Test(): HF_Failed_Model_Test_Fix0("p208p2002/bart-drcd-qg-hl")
def HF_Failed_Model_1_Test(): HF_Failed_Model_Test_Fix0("p208p2002/bart-drcd-qg-hl-v2")
def HF_Failed_Model_2_Test(): HF_Failed_Model_Test_Fix0("naem1023/bart-v2-speech")
def HF_Failed_Model_3_Test(): HF_Failed_Model_Test_Fix0("uer/bart-large-chinese-cluecorpussmall")
def HF_Failed_Model_4_Test(): HF_Failed_Model_Test_Fix0("beyond/genius-base-chinese")
def HF_Failed_Model_5_Test(): HF_Failed_Model_Test_Fix0("nlpotato/kobart_chatbot_social_media-e10_1")
def HF_Failed_Model_6_Test(): HF_Failed_Model_Test_Fix0("uer/pegasus-base-chinese-cluecorpussmall")
def HF_Failed_Model_7_Test(): HF_Failed_Model_Test_Fix2("facebook/detr-resnet-50")
def HF_Failed_Model_8_Test(): HF_Failed_Model_Test_Fix3("abhilash1910/albert-german-ner")
def HF_Failed_Model_9_Test(): HF_Failed_Model_Test_Fix4("microsoft/tapex-base-finetuned-wikisql")
def HF_Failed_Model_10_Test(): HF_Failed_Model_Test_Fix5("speechbrain/m-ctc-t-large")
def HF_Failed_Model_11_Test(): HF_Failed_Model_Test_Fix6("anton-l/distilhubert-ft-keyword-spotting")

if __name__ == "__main__":

    #RNN_test()
    #MLP_Test()
    #print('===')

    #ResNet50_Test()
    #ResNet18_Test()
    #Order_Test()
    #ResNet18_Onnx_Test()
    Custom_Test()
    #ResNet101_Onnx_Test()
    #AN_TORCHONNX_Val_Test()
    #ResNet101_comp_Test()
    #SkipConnection_Test()
    #Hash_Order_Test()
    #ResNet50_Hash_Comp_Test()
    #ResNet101_Hash_Comp_Test()
    #BigNet_Hash_Comp_Test()
    #BigResNet_Hash_Comp_Test()
    #HugeNet_Hash_Comp_Test()
    #ParallelNet_Hash_Comp_Test()
    #Vector_ResNet50_Test()
    #Custom3_Test()
    #HF_Failed_Model_11_Test()