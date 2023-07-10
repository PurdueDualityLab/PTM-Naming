
from transformers import ResNetForImageClassification
from comp import OrderedListComparator
from utils import print_list
import timm
import torch

m1 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
m2 = timm.create_model('resnet50.a1_in1k', pretrained=True)
m2.eval()
comp = OrderedListComparator()
comp.from_model(m1, m2, torch.zeros(1, 3, 224, 224))
#print_list(comp.l1)
#print('======')
#print_list(comp.l2)
print(comp.get_diff())
