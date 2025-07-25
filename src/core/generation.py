import torch
import torch.nn as nn
from dataclasses import dataclass

from core.tokenizer import Tokenizer

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    maximum_length: float
    temperature: float
    device: torch.device

def generate_text(model: nn.Module, tokenizer: Tokenizer, prompt: str, config: GenerationConfig) -> str:
    """Generate text based on a prompt"""
    token_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(config.device)

    for _ in range(int(config.maximum_length)):
        logits = model(token_ids)
        logits = logits[0, -1, :]
        logits = logits / config.temperature

        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(token_ids[0].cpu().tolist())
