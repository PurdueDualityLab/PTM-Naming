
from ANN.AbstractNN import generate_ordered_layer_list_from_pytorch_model, generate_ordered_layer_list_from_onnx_model, generate_ordered_layer_list_from_pytorch_model_with_id_and_connection
from transformers import AutoModel, AutoTokenizer
import difflib
import onnx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import numpy as np
from ANN.ann_layer import AbstractNNLayer

from ANN.utils import overwrite_torchview_func

class ANNComparator():

    def __init__(self, l1=None, l2=None, c1=None, c2=None):
        self.l1 = l1
        self.l2 = l2
        self.c1 = c1
        self.c2 = c2

    def from_model(self, m1, m2, inp):
        overwrite_torchview_func()
        self.l1 = generate_ordered_layer_list_from_pytorch_model(m1, inp)
        self.l2 = generate_ordered_layer_list_from_pytorch_model(m2, inp)
    
    def from_NLP_model_name(self, n1, n2):
        overwrite_torchview_func()
        t = AutoTokenizer.from_pretrained(n1)
        m1 = AutoModel.from_pretrained(n1)
        inp = t("Test Input", return_tensors="pt")
        ci1 = generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(m1, inp)
        self.l1 = ci1[0]
        self.c1 = ci1[1]
        m2 = AutoModel.from_pretrained(n2)
        ci2 = generate_ordered_layer_list_from_pytorch_model_with_id_and_connection(m2, inp)
        self.l2 = ci2[0]
        self.c2 = ci2[1]
    
    def from_ONNX_model_directory(self, dir1, dir2):
        m1 = onnx.load(dir1)
        m2 = onnx.load(dir2)
        self.l1 = generate_ordered_layer_list_from_onnx_model(m1)
        self.l2 = generate_ordered_layer_list_from_onnx_model(m2)

    def get_diff(self):
        
        str_l1 = [str(e) for e in self.l1]
        str_l2 = [str(e) for e in self.l2]
        '''
        i1, i2 = 0, 0

        while i1 != len(str_l1) and i2 != len(str_l2):
            if str_l1[i1] == str_l2[i2]:
                print(str_l1[i1])
                i1 += 1
                i2 += 1
            elif str_l1[i1] in str_l2[i2:]:
                print('+ ' + str_l1[i1])
                i1 += 1
            elif str_l2[i2] in str_l1[i1:]:
                print('- ' + str_l2[i2])
                i2 += 1
            else: # Element in neither list ahead, move both pointers
                print('+ ' + str_l1[i1])
                print('- ' + str_l2[i2])
                i1 += 1
                i2 += 1

        # For remaining elements in either list
        while i1 < len(str_l1):
            print('+ ' + str_l1[i1])
            i1 += 1
        while i2 < len(str_l2):
            print('- ' + str_l2[i2])
            i2 += 1
        return None'''
        differ: difflib.Differ = difflib.Differ()
        diff = differ.compare(str_l1, str_l2)
        
        for i in diff:
            print(i)

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

    def get_ngram_cosine_similarity(self, ngram_range=(50, 50)):
        
        str_l1 = [str(e) for e in self.l1]
        str_l2 = [str(e) for e in self.l2]

        str1 = ' '.join(str_l1)
        str2 = ' '.join(str_l2)
        
        vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range).fit([str1, str2])
        
        ngrams1 = vectorizer.transform([str1])
        ngrams2 = vectorizer.transform([str2])
        
        similarity = cosine_similarity(ngrams1, ngrams2)
        
        return similarity[0][0]

    def get_nlayer_cosine_similarity(self) -> int:
        if len(self.c1) != len(self.l1) or len(self.c2) != len(self.l2):
            print('connection info not matched')
            return None

        def get_freq_vec(layer_list, connection_info):
            freq_vec = dict()
            id_to_node_map = dict()
            for layer_node, layer_connection_info in zip(layer_list, connection_info): # assume no repetitive layer in ordered list
                id_to_node_map[layer_connection_info[0]] = layer_node
                
            def make_node_string(n: AbstractNNLayer):
                if n.is_input_node:
                    return '[INPUT]'
                if n.is_output_node:
                    return '[OUTPUT]'
                return '{} {}->{}'.format(n.operation, n.input_shape, n.output_shape)

            for layer_node, layer_connection_info in zip(layer_list, connection_info):
                curr_node_str = make_node_string(layer_node)
                for next_layer_id in layer_connection_info[1]:
                    next_node_str = make_node_string(id_to_node_map[next_layer_id])
                    combined_str = '({}, {})'.format(curr_node_str, next_node_str)
                    if combined_str not in freq_vec:
                        freq_vec[combined_str] = 0
                    freq_vec[combined_str] += 1
            
            return freq_vec
            
        fv1 = get_freq_vec(self.l1, self.c1)
        fv2 = get_freq_vec(self.l2, self.c2)

        keys = list(set(list(fv1.keys()) + list(fv2.keys())))

        vec1 = np.array([fv1.get(key, 0) for key in keys])
        vec2 = np.array([fv2.get(key, 0) for key in keys])

        cos_sim = cosine_similarity([vec1], [vec2])

        return cos_sim[0][0]
    
    def get_param_cosine_similarity(self) -> int:

        def get_freq_vec(layer_list: List[AbstractNNLayer]):
            freq_vec = dict()
            for l in layer_list:
                p_str_list = []
                if l.parameters != None:
                    for p in l.parameters:
                        p_str_list.append('<{}, {}>'.format(p.param_name, p.param_value))
                if l.is_input_node:
                    l_str = '[INPUT]'
                elif l.is_output_node:
                    l_str = '[OUTPUT]'
                else:
                    l_str = '{} {}'.format(l.operation, p_str_list if len(p_str_list) else '')
                if l_str not in freq_vec:
                    freq_vec[l_str] = 0
                freq_vec[l_str] += 1
            return freq_vec
        
        fv1 = get_freq_vec(self.l1)
        fv2 = get_freq_vec(self.l2)

        for k, v in fv1.items():
            print(k, v)
        print('\n-=-=-=-=\n')
        for k, v in fv2.items():
            print(k, v)

        keys = list(set(list(fv1.keys()) + list(fv2.keys())))

        vec1 = np.array([fv1.get(key, 0) for key in keys])
        vec2 = np.array([fv2.get(key, 0) for key in keys])

        cos_sim = cosine_similarity([vec1], [vec2])

        return cos_sim[0][0]
