import torch
import torch.nn as nn

from core.tokenizer import Tokenizer
from core.model import ModelConfig, create_model

def save_model(config: ModelConfig, model: nn.Module, filename: str):
    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    torch.save({
        'model_state_dict': state_dict,
        'config': config.__dict__
    }, filename)

def load_model(config: ModelConfig, tokenizer: Tokenizer, filename: str) -> nn.Module:
    saved_model = torch.load(filename, weights_only=False)
    
    if 'config' in saved_model:
        for key, value in saved_model['config'].items():
            setattr(config, key, value)
        state_dict = saved_model['model_state_dict']
    else:
        state_dict = saved_model
    
    model = create_model(config, tokenizer.vocabulary_size)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = value
        model.load_state_dict(clean_state_dict)

    return model
