import torch
import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from typing import Tuple
from .tokenizer import Tokenizer
from .model import ModelConfig

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.data[idx]["input_ids"]
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets

def download_text_dataset(dataset_name: str, dataset_sub: str, tokenizer: Tokenizer, model_config: ModelConfig, processor_number: int):
    dataset = load_dataset(dataset_name, dataset_sub, split="train", num_proc=processor_number)
    split_dataset = dataset.train_test_split(test_size=0.01, shuffle=True)
    training_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    def tokenize(example):
        return {"input_ids": [tokenizer.encode(text) for text in example["text"]]}
    
    training_data_tokenized = training_dataset.map(tokenize, batched=True, num_proc=processor_number, remove_columns=["text"])
    validation_data_tokenized = validation_dataset.map(tokenize, batched=True, num_proc=processor_number, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // model_config.block_size) * model_config.block_size
        result = {k: [t[i:i+model_config.block_size] for i in range(0, total_length, model_config.block_size)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result
    
    training_data_chunked = training_data_tokenized.map(group_texts, batched=True, remove_columns=training_data_tokenized.column_names, num_proc=processor_number)
    validation_data_chunked = validation_data_tokenized.map(group_texts, batched=True, remove_columns=training_data_tokenized.column_names, num_proc=processor_number)

    training_data_chunked.save_to_disk(f"data/{dataset_name}-{dataset_sub}-training-tokenized")
    validation_data_chunked.save_to_disk(f"data/{dataset_name}-{dataset_sub}-validation-tokenized")

def load_text_dataset(dataset_name: str,dataset_sub: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    training_dataset = load_from_disk(f"data/{dataset_name}-{dataset_sub}-training-tokenized")
    validation_dataset = load_from_disk(f"data/{dataset_name}-{dataset_sub}-validation-tokenized")

    training_dataset.set_format(type="torch", columns=["input_ids"])
    validation_dataset.set_format(type="torch", columns=["input_ids"])

    train_dataset = TextDataset(training_dataset)
    val_dataset = TextDataset(validation_dataset)

    training_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    return training_dataloader, validation_dataloader

def dataset_exists(dataset_name: str, dataset_sub: str) -> bool:
    training_path = f"data/{dataset_name}-{dataset_sub}-training-tokenized"
    validation_path = f"data/{dataset_name}-{dataset_sub}-validation-tokenized"
    return os.path.exists(training_path) and os.path.exists(validation_path)
