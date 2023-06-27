
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
        str_l1 = [str(e) for e in self.l1]
        str_l2 = [str(e) for e in self.l2]
        differ: difflib.Differ = difflib.Differ()
        diff = differ.compare(str_l1, str_l2)
        for i in diff:
            print(i)

        return diff

    def get_sim_metric(self): # not a good metric
        matcher = difflib.SequenceMatcher(None, self.l1, self.l2)
        return matcher.ratio()
    
    def get_approximate_ged(self):
        diff = self.get_diff()
        ged = 0
        for i in diff:
            if i[0] in {'-', '+'}:
                ged += 1
        return ged
