import json
import random

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class transformer_dataset(Dataset):
    def __init__(self, dict_path: str, label_type: str, tokenizer, trim) -> None:
        self.label_type = label_type
        self.model_dict = self.load_dict(dict_path)  # Load model vectors
        self.model_names = list(self.model_dict.keys())  # List of model names
        # self.num_classes = len(set(self.model_dict[model_name][label_type] for model_name in self.model_names))  # Number of unique classes
        self.tokenizer = tokenizer
        self.max_length = min(tokenizer.model_max_length, 4096)
        # self.trim = min(trim, int(self.max_length * 0.75))   # default is 3:1 ratio
        self.trim = int(self.max_length * 0.75)
        print(f"trim: {self.trim}")
        self.data = self.processing()
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([item['labels'] for item in self.data])
        self.label_to_index = {label: idx for idx, label in enumerate(self.mlb.classes_)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}  # Create index to label mapping

        if self.label_type == 'task':
            self.convert_labels_to_binary()
        else:
            self.convert_labels_to_indices()  # Convert labels to indices
        self.shuffle_data()  # Shuffle data for training
        

    def load_dict(self, dict_path: str) -> dict:
        with open(dict_path, 'r') as f:
            model_dict = json.load(f)
        return model_dict


    def processing(self) -> list:
        data = []
        avg_length = 0
        for model_name in self.model_names:
            layers = self.model_dict[model_name]['layers']
            label = self.model_dict[model_name][self.label_type]    #this could be multiple labels
            if not isinstance(label, list):
                label = [label]
            
            tokenized = self.tokenizer(layers, padding=False, truncation=False)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']

            #try without it
            avg_length += len(input_ids)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.trim] + input_ids[-(self.max_length - self.trim):]
                attention_mask = attention_mask[:self.trim] + attention_mask[-(self.max_length - self.trim):]
            # Pad sequences shorter than max_length.
            if len(input_ids) < self.max_length:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                attention_mask = attention_mask + [0] * (self.max_length - len(attention_mask))

            # global_attention_mask = [0] * self.max_length  # Default: all local attention
            # global_attention_mask[0] = 1 
        
            data.append({
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                # "global_attention_mask": global_attention_mask,
                "labels": label
            })
        avg_length /= len(self.model_names)
        print(f"Average length of input_ids: {avg_length}")
        return data
    
    # def create_label_mapping(self):
    #     unique_labels = sorted(set(item["labels"] for item in self.data)) 
    #     return {label: idx for idx, label in enumerate(unique_labels)}

    def convert_labels_to_binary(self):
        for data in self.data:
            data['labels'] = self.mlb.transform([data['labels']])[0].tolist()
        
    def convert_labels_to_indices(self):
        self.data = [{**item, "labels": self.label_to_index[item["labels"][0]]} for item in self.data]

    def shuffle_data(self):
        random.shuffle(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.label_type == 'task':
            label = torch.tensor(item["labels"], dtype=torch.float)
        else:
            label = torch.tensor(item["labels"], dtype=torch.long)
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            # "global_attention_mask": torch.tensor(item["global_attention_mask"], dtype=torch.long),
            "labels": label
        }

    def __len__(self):
        return len(self.data)

    
    def train_len(self):
        return len(self.train_data)

    
    def test_len(self):
        return len(self.test_data)

    
    def get_num_classes(self):
        return len(self.mlb.classes_)


    def get_label_mapping(self):
        return self.index_to_label
    
    def get_index_mapping(self):
        return self.label_to_index