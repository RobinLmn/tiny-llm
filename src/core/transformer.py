import torch
import torch.nn as nn

from core.attention import MultiHeadAttention
from core.embeddings import Embeddings

class FeedForward(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, embedding_dimension: int, hidden_dimension: int):
        super().__init__()
        self.first_layer = nn.Linear(embedding_dimension, hidden_dimension)
        self.second_layer = nn.Linear(hidden_dimension, embedding_dimension)
        self.relu = nn.ReLU()

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward layer"""
        token_embeddings = self.first_layer(token_embeddings)
        token_embeddings = self.relu(token_embeddings)
        token_embeddings = self.second_layer(token_embeddings)
        return token_embeddings
    
class TransformerBlock(nn.Module):
    """Transformer Block Layers"""
    def __init__(self, embedding_dimension: int, head_number: int, hidden_dimension: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dimension, head_number)
        self.feed_forward = FeedForward(embedding_dimension, hidden_dimension)
        self.attention_normalization_layer = nn.LayerNorm(embedding_dimension)
        self.feed_forward_normalization_layer = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with attention and feed-forward layers."""
        attention_output = self.multi_head_attention(self.attention_normalization_layer(token_embeddings))
        token_embeddings = token_embeddings + self.dropout(attention_output)
        
        feed_forward_output = self.feed_forward(self.feed_forward_normalization_layer(token_embeddings))
        token_embeddings = token_embeddings + self.dropout(feed_forward_output)
        
        return token_embeddings

class TinyLLM(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dimension: int, max_sequence_length: int, layer_number: int, head_number: int, hidden_dimension: int, device: torch.device):
        super().__init__()
        self.embeddings = Embeddings(vocabulary_size, embedding_dimension, max_sequence_length)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embedding_dimension, head_number, hidden_dimension) for _ in range(layer_number)])
        self.normalization_layer = nn.LayerNorm(embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, vocabulary_size)
        self.to(device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete transformer model."""
        token_embeddings = self.embeddings(token_ids)
        for transformer_block in self.transformer_blocks:
            token_embeddings = transformer_block(token_embeddings)
        return self.output_projection(self.normalization_layer(token_embeddings))
