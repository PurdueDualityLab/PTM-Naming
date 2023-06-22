
import torch
import torch.nn as nn
from list_gen import OrderedListGenerator
from transformers import ResNetForImageClassification

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
    
def RNN_test():
    gen = OrderedListGenerator(TestRNN(), torch.randn(2, 3))
    gen.print_connection()

def MLP_Test():
    gen = OrderedListGenerator(TestMLP(), torch.randn(2, 128))
    gen.print_connection()

def ResNet18_Test():
    rn18 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
    gen = OrderedListGenerator(rn18, torch.randn(1, 3, 224, 224))
    gen.print_connection()

def ResNet50_Test():
    rn50 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    gen = OrderedListGenerator(rn50, torch.randn(1, 3, 224, 224))
    gen.print_connection()

def Order_Test():
    orderA = OrderATest()
    orderB = OrderBTest()
    i = (torch.randn(1, 10))
    genA = OrderedListGenerator(orderA, i)
    genA.print_ordered_list()
    print()
    genB = OrderedListGenerator(orderB, i) 
    genB.print_ordered_list()


#RNN_test()
#MLP_Test()
#ResNet50_Test()
ResNet18_Test()
##Order_Test()

