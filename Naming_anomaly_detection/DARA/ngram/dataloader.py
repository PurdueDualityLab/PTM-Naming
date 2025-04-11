import json
import random
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer

class DARA_dataset(Dataset):
    def __init__(self, dict_path: str, label_type: str, train_ratio: float = 0.8) -> None:
        self.label_type = label_type
        self.model_dict = self.load_dict(dict_path)
        self.model_names = list(self.model_dict.keys())
        self.data = self.processing()
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([item[1] for item in self.data])
        self.label_to_index = {label: idx for idx, label in enumerate(self.mlb.classes_)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        if label_type == 'task':
            self.convert_labels_to_binary()
        else:
            self.convert_labels_to_indices()
        self.shuffle_data()

    def load_dict(self, dict_path: str) -> dict:
        with open(dict_path, 'r') as f:
            return json.load(f)

    def processing(self) -> list:
        data = []
        for model_name in self.model_names:
            # using both l and p
            # l_tensor = torch.tensor(self.model_dict[model_name]['l'], dtype=torch.float).unsqueeze(0)
            # p_tensor = (1 * torch.tensor(self.model_dict[model_name]['p'], dtype=torch.float)).unsqueeze(0)
            # vec = torch.cat((l_tensor, p_tensor), dim=1)
            
            # using only l
            vec = torch.tensor(self.model_dict[model_name]['l'], dtype=torch.float).unsqueeze(0)
            label = self.model_dict[model_name][self.label_type]
            if not isinstance(label, list):
                label = [label]  # Ensure labels are always lists
            data.append((vec, label, model_name))
        return data

    def convert_labels_to_binary(self):
        new_data = []
        for vec, labels, name in self.data:
            binary_labels = self.mlb.transform([labels])[0].tolist()
            new_data.append((vec, binary_labels, name))
        self.data = new_data
    
    def convert_labels_to_indices(self):
        self.data = [(vec, self.label_to_index[label[0]], name) for vec, label, name in self.data]

    def shuffle_data(self):
        random.shuffle(self.data)

    def __getitem__(self, idx):
        vec, label, name = self.data[idx]  # Adjust if your data structure is different
        if self.label_type == 'task':
            label = torch.tensor(label, dtype=torch.float)
        return vec, label, name
    
    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return len(self.mlb.classes_)

    def get_label_mapping(self):
        return self.index_to_label

    def get_data_shape(self):
        return self.data[0][0].size()