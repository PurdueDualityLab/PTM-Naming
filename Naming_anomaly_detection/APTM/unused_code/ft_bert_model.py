from transformers import BertTokenizer, AutoModel, Trainer, TrainingArguments
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import json

with open("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch/layer_name_data.json", "r") as f:
    sentences = json.load(f)

tokenizer = BertTokenizer("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch/vocab.txt")

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print(inputs)

dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])

print(dataset)

data_loader = DataLoader(dataset, batch_size=32)

model = AutoModel.from_pretrained('bert-base-uncased')

training_args = TrainingArguments(
    per_device_train_batch_size=32,
    num_train_epochs=3,
    output_dir="/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("/depot/davisjam/data/chingwo/PTM-Naming/vectorizer/pytorch")