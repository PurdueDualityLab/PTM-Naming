import torch
import torch.utils.data as data
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd

from loguru import logger

class Huggingface(data.Dataset):
    def __init__(self, args, root, tokenizer, test=False, trim=256):
        self.args = args
        self.root = root
        self.train = pd.read_json(self.root + '/PeaTMOSS_data/balanced_dataset/train_dataset.json')
        self.df = pd.read_json(self.root + '/PeaTMOSS_data/balanced_dataset/test_dataset.json') if test else pd.read_json(self.root + '/PeaTMOSS_data/balanced_dataset/train_dataset.json')
        self.tokenizer = tokenizer
        self.trim = trim
        
        self.tokens = [self.tokenization(i) for i in self.df["layers"]]
        
        if args.train_mode == 'model_type':
            self.labels = self.train.model_type
            self.df_labels = self.df.model_type.to_list()
        elif args.train_mode == 'task':
            self.labels = self.train.task
            self.df_labels = self.df.task.to_list()
        elif args.train_mode == 'arch':
            self.labels = self.train.arch
            self.df_labels = self.df.arch.to_list()
        else:
            logger.error("Invalid label type")
            raise ValueError("Invalid label type")

        self.label_to_index = self.create_label_mapping()
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}        
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        '''
        returns ith index of layer embeddings, architecture, task types. 
        '''
        return self.tokens[idx], self.label_to_index[self.df_labels[idx]]
        
    def tokenization(self, layers):
        '''
        returns tokenizer object with input_ids and attention_mask
        '''
        if self.args.model_name == 'longformer':
            return self.tokenizer(layers, padding='max_length', truncation=True, max_length=1024, return_tensors="pt")
        if self.trim == 512:
            return self.tokenizer(layers, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        inputs = self.tokenizer(layers, return_tensors="pt")
        
        _input_ids = inputs["input_ids"].squeeze().tolist()
        _attention_mask = inputs["attention_mask"].squeeze().tolist()
        
        # TODO: check if this is valid for both if and else.
        
        if len(_input_ids) > 512:
            input_ids = _input_ids[:self.trim] + _input_ids[-(512-self.trim):]
            attention_mask = _attention_mask[:self.trim] + _attention_mask[-(512-self.trim):]
        else: #TODO: FIX
            half = int(self.trim / 512 * len(_input_ids)) # same ratio as trim
            input_ids = _input_ids[:half] + [1] * (512-len(_input_ids)) + _input_ids[-(len(_input_ids)-half):]
            attention_mask = _attention_mask[:half] + [0] * (512-len(_attention_mask)) + _attention_mask[-(len(_attention_mask)-half):]
            
        inputs["input_ids"] = torch.tensor([input_ids])
        inputs["attention_mask"] = torch.tensor([attention_mask])
        
        return inputs
    
    def tokenization_test(self, layers):
        return self.tokenizer(layers, return_tensors="pt")
    
    def create_label_mapping(self):
        unique_labels = sorted(set(label for label in self.labels))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
if __name__ == '__main__':
    model = RobertaModel.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = Huggingface(root='/depot/davisjam/data/mingyu/ptm-contrastive-learning/contrastive_learning', tokenizer=tokenizer)
    
    # Test the dataset
    sample_idx = 3
    sample_data = train_dataset[sample_idx]
    tokens, arch_class, task_class = sample_data

    print("Tokens:", tokens)
    print("Architecture Class:", arch_class)
    print("Task Class:", task_class)
    print(len(tokens["input_ids"][0]), len(tokens["attention_mask"][0]))
    
    print("Token shape: ", tokens.shape)
    