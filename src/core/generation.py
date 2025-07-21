import torch
import torch.nn as nn

from core.tokenizer import Tokenizer

def generate_text(model: nn.Module, tokenizer: Tokenizer, prompt: str, max_length: int, temperature: float, device: torch.device) -> str:
    """Generate text based on a prompt"""
    token_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    for _ in range(max_length):
        logits = model(token_ids)
        logits = logits[0, -1, :]
        logits = logits / temperature

        probabilities = torch.softmax(logits , dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

    return tokenizer.decode(token_ids[0].cpu().tolist())
