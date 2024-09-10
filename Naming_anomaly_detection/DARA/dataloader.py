import json
import random
import torch

class DARA_dataset:
    def __init__(self, dict_path: str, label_type: str) -> None:
        self.label_type = label_type
        self.model_dict = self.load_dict(dict_path)  # Load model vectors
        self.model_names = list(self.model_dict.keys())  # List of model names
        self.num_classes = len(set(self.model_dict[model_name]['arch'] for model_name in self.model_names))  # Number of unique classes
        self.data = self.processing()  # Merge vectors with labels
        self.label_to_index = self.create_label_mapping()  # Create label mapping
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}  # Create index to label mapping

        self.convert_labels_to_indices()  # Convert labels to indices
        self.shuffle_data()  # Shuffle data for training
        self.train_data, self.test_data = self.split_data()  # Split data into training and testing sets

    def load_dict(self, dict_path: str) -> dict:
        with open(dict_path, 'r') as f:
            model_dict = json.load(f)
        return model_dict


    def processing(self) -> list:
        data = []
        for model_name in self.model_names:
            '''
            TODO: Check different weights and settings of the vectors.
            '''
            l_tensor = torch.tensor(self.model_dict[model_name]['l'], dtype=torch.float).unsqueeze(0)  # Add an extra dimension
            p_tensor = (1 * torch.tensor(self.model_dict[model_name]['p'], dtype=torch.float)).unsqueeze(0)  # Scale and add an extra dimension
            vec = torch.cat((l_tensor, p_tensor), dim=1)  # Concatenate along the new dimension

            # torch.tensor(self.model_dict[model_name]['l'], dtype=torch.float) + 0.5*torch.tensor(self.model_dict[model_name]['p'], dtype=torch.float)
            label = self.model_dict[model_name][self.label_type]
            data.append((vec, label, model_name))
        return data

    def create_label_mapping(self):
        unique_labels = sorted(set(label for _, label, _ in self.data))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def convert_labels_to_indices(self):
        self.data = [(vec, self.label_to_index[label], name) for vec, label, name in self.data]

    def shuffle_data(self):
        random.shuffle(self.data)

    def split_data(self, train_ratio: float = 0.8):
        split_index = int(len(self.data) * train_ratio)
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]
        return train_data, test_data


    def __getitem__(self, idx):
        # Get the data sample (features and label) for the given index
        vec, label, name = self.data[idx]  # Adjust if your data structure is different
        return vec, label, name

    def __len__(self):
        return len(self.data)

    
    def train_len(self):
        return len(self.train_data)

    
    def test_len(self):
        return len(self.test_data)

    
    def get_num_classes(self):
        return self.num_classes


    def get_label_mapping(self):
        return self.index_to_label


    def get_data_shape(self):
        return self.data[0][0].size()