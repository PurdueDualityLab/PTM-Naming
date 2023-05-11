from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
import LayerClass

#processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
model2 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

np1 = list(model1.named_parameters())
np2 = list(model2.named_parameters())

for i in range(len(np1)):
    print(np1[i][0], np2[i][0])
    print(np1[i][1].size(), np2[i][1].size())
'''
l = LayerClass.LayerTools.CreateFromString('Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)')
print(l)'''