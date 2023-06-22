
from utils import generate_ordered_layer_list_from_pytorch_model, patch
import difflib

class OrderedListComparator():

    def __init__(self, l1=None, l2=None):
        self.l1 = l1
        self.l2 = l2

    def from_model(self, m1, m2, inp):
        patch()
        self.l1 = generate_ordered_layer_list_from_pytorch_model(m1, inp)
        self.l2 = generate_ordered_layer_list_from_pytorch_model(m2, inp)

    def get_diff(self):
        matcher = difflib.SequenceMatcher(None, self.l1, self.l2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'delete':
                for i in range(i1, i2):
                    print(f"- {self.l1[i]}")
            elif tag == 'insert':
                for i in range(j1, j2):
                    print(f"+ {self.l2[i]}")

    def get_sim_metric(self):
        matcher = difflib.SequenceMatcher(None, self.l1, self.l2)
        return matcher.ratio()
