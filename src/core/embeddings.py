import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """Embedding Layers"""
    def __init__(self, vocabulary_size: int, embedding_dimension: int, max_sequence_length: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_embeddings = nn.Embedding(max_sequence_length, embedding_dimension)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply token and positional embeddings to token IDs"""
        sequence_length = token_ids.size(-1)
        positions = torch.arange(sequence_length, device=token_ids.device)

        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.positional_embeddings(positions)
        return token_embeddings + position_embeddings
