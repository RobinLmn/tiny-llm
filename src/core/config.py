import torch
from dataclasses import dataclass
from typing import Tuple

from core.tokenizer import Tokenizer
from core.transformer import TinyLLM

@dataclass
class ModelConfig:
    """Configuration for a TinyLLM model and training"""
    sequence_length: int = 64
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    embedding_dimension: int = 128
    head_number: int = 4
    hidden_dimension: int = 256
    layer_number: int = 4
    max_sequence_length: int = 128
    max_generated_length: int = 50
    temperature: float = 0.8
    training_data_file: str = 'data/shakespeare.txt'

def save_model(config: ModelConfig, model: TinyLLM, tokenizer: Tokenizer, filename: str) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocabulary,
        'tokenizer_inverse_vocab': tokenizer.inverted_vocabulary,
        'config': config.__dict__
    }, filename)

def load_model(config: ModelConfig, filename: str, device: torch.device) -> Tuple[TinyLLM, Tokenizer]:
    """Load a saved model and tokenizer"""
    checkpoint = torch.load(filename, map_location=device)

    for key, value in checkpoint['config'].items():
        setattr(config, key, value)

    tokenizer = Tokenizer()
    tokenizer.vocabulary = checkpoint['tokenizer_vocab']
    tokenizer.inverted_vocabulary = checkpoint['tokenizer_inverse_vocab']
    
    model = create_model(config, tokenizer.vocabulary_size, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer

def create_model(config: ModelConfig, vocabulary_size: int, device: torch.device):
    """Create a model with the given config"""
    return TinyLLM(
        vocabulary_size=vocabulary_size,
        embedding_dimension=config.embedding_dimension,
        max_sequence_length=config.max_sequence_length,
        layer_number=config.layer_number,
        head_number=config.head_number,
        hidden_dimension=config.hidden_dimension,
        device=device
        )
    