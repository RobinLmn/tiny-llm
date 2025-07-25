import torch
import torch.nn as nn
from typing import Tuple

from core.tokenizer import Tokenizer
from core.model import ModelConfig, create_model

def save_model(config: ModelConfig, model: nn.Module, tokenizer: Tokenizer, filename: str):
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_type': tokenizer.tokenizer_type,
        'tokenizer_vocab': tokenizer.vocabulary,
        'tokenizer_inverse_vocab': tokenizer.inverted_vocabulary,
        'config': config.__dict__
    }, filename)

def load_model(config: ModelConfig, filename: str) -> Tuple[nn.Module, Tokenizer]:
    saved_model = torch.load(filename, weights_only=False)
    for key, value in saved_model['config'].items():
        setattr(config, key, value)

    tokenizer = Tokenizer(saved_model['tokenizer_type'])
    tokenizer.vocabulary = saved_model['tokenizer_vocab']
    tokenizer.inverted_vocabulary = saved_model['tokenizer_inverse_vocab']
    
    model = create_model(config, tokenizer.vocabulary_size)
    model.load_state_dict(saved_model['model_state_dict'])
    model.eval()

    return model, tokenizer
