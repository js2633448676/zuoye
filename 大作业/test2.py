import json
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader

# Load data
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

train_data = load_data('train.json')
dev_data = load_data('dev.json')
test_data = load_data('test.json')

# Define your custom Dataset
class SemanticMatchingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']

        encoding = self.tokenizer(premise, hypothesis, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

# Define model architecture
model_name = "bert-base-uncased"  # You can choose a different model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Adjust num_labels according to your task

# Create Datasets and DataLoaders
max_length = 128  # You can adjust this as needed
train_dataset = SemanticMatchingDataset(train_data, tokenizer, max_length)
dev_dataset = SemanticMatchingDataset(dev_data, tokenizer, max_length)
test_dataset = SemanticMatchingDataset(test_data, tokenizer, max_length)

batch_size = 32  # You can adjust this as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)  # You can adjust the learning rate

num_epochs = 5  # You can adjust this
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    # ... perform validation on the dev set ...

# Testing loop
model.eval()
# ... perform testing on the test set ...
