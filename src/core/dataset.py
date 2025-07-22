import torch
from torch.utils.data import Dataset

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
