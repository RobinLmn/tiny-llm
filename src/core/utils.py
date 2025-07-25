import torch
import torch.nn as nn
from typing import Tuple

from core.tokenizer import Tokenizer
from core.model import ModelConfig, create_model

def save_model(config: ModelConfig, model: nn.Module, filename: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__
    }, filename)

def load_model(config: ModelConfig, tokenizer: Tokenizer, filename: str) -> nn.Module:
    saved_model = torch.load(filename, weights_only=False)
    for key, value in saved_model['config'].items():
        setattr(config, key, value)
    
    model = create_model(config, tokenizer.vocabulary_size)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    return model
