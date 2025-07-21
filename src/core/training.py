import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from core.tokenizer import Tokenizer

def prepare_sequences(text: str, tokenizer: Tokenizer, sequence_length: int):
    """Convert text into input-target pairs for next-token prediction"""
    tokens = tokenizer.encode(text)
    inputs, targets = [], []
    for i in range(len(tokens) - sequence_length):
        inputs.append(tokens[i:i + sequence_length])
        targets.append(tokens[i + 1:i + sequence_length + 1])
    return inputs, targets

class TextDataset(Dataset):
    """Text training dataset"""
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def train_model(model: nn.Module, dataloader: DataLoader, epochs: int, learning_rate: float, device: torch.device):
    """Train the model over a dataset"""
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            if batch % 1000 == 0:
                print(f'Epoch {epoch}, Batch {batch}, Loss: {loss.item():.4f}')
