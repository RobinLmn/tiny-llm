import torch
import torch.nn as nn
from dataclasses import dataclass

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layers"""
    def __init__(self, embedding_dimension: int, head_number: int, dropout: float, block_size: int):
        super().__init__()
        self.attention_projection = nn.Linear(embedding_dimension, 3 * embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.residual_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        self.head_number = head_number
        self.embedding_dimension = embedding_dimension

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply multi-head attention to token embeddings"""
        batch_size, sequence_length, embedding_dimension = token_embeddings.size()

        query, key, value = self.attention_projection(token_embeddings).split(self.embedding_dimension, dim=2)

        head_dimension = embedding_dimension // self.head_number
        query = query.view(batch_size, sequence_length, self.head_number, head_dimension).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.head_number, head_dimension).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.head_number, head_dimension).transpose(1, 2)

        attention = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dimension)

        return self.residual_dropout(self.output_projection(attention))

class Embeddings(nn.Module):
    """Embedding Layers"""
    def __init__(self, vocabulary_size: int, embedding_dimension: int, block_size: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_embeddings = nn.Embedding(block_size, embedding_dimension)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Apply token and positional embeddings to token IDs"""
        sequence_length = token_ids.size(-1)
        positions = torch.arange(sequence_length, device=token_ids.device)

        token_embeddings = self.token_embeddings(token_ids)
        position_embeddings = self.positional_embeddings(positions)
        return token_embeddings + position_embeddings

class FeedForward(nn.Module):
    """Feed-Forward Neural Network"""
    def __init__(self, embedding_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.first_layer = nn.Linear(embedding_dimension, hidden_dimension)
        self.second_layer = nn.Linear(hidden_dimension, embedding_dimension)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward layer"""
        token_embeddings = self.first_layer(token_embeddings)
        token_embeddings = self.gelu(token_embeddings)
        token_embeddings = self.second_layer(token_embeddings)
        token_embeddings = self.dropout(token_embeddings)
        return token_embeddings
    
class TransformerBlock(nn.Module):
    """Transformer Block Layers"""
    def __init__(self, embedding_dimension: int, head_number: int, hidden_dimension: int, dropout: float, block_size: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dimension, head_number, dropout, block_size)
        self.feed_forward = FeedForward(embedding_dimension, hidden_dimension, dropout)
        self.attention_normalization_layer = nn.LayerNorm(embedding_dimension)
        self.feed_forward_normalization_layer = nn.LayerNorm(embedding_dimension)

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply transformer block with attention and feed-forward layers"""
        token_embeddings = token_embeddings + self.multi_head_attention(self.attention_normalization_layer(token_embeddings))
        token_embeddings = token_embeddings + self.feed_forward(self.feed_forward_normalization_layer(token_embeddings))
        return token_embeddings

@dataclass
class ModelConfig:
    """Configuration for a model"""
    embedding_dimension: int
    block_size: int
    layer_number: int
    head_number: int
    hidden_dimension: int
    dropout: float
    device: torch.device

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, vocabulary_size: int):
        super().__init__()
        self.embeddings = Embeddings(vocabulary_size, config.embedding_dimension, config.block_size)
        self.dropout = nn.Dropout(config.dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config.embedding_dimension, config.head_number, config.hidden_dimension, config.dropout, config.block_size) for _ in range(config.layer_number)
        ])
        self.normalization_layer = nn.LayerNorm(config.embedding_dimension)
        self.output_projection = nn.Linear(config.embedding_dimension, vocabulary_size, bias=False)

        self.to(config.device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through the complete transformer model."""
        token_embeddings = self.dropout(self.embeddings(token_ids))
        for transformer_block in self.transformer_blocks:
            token_embeddings = transformer_block(token_embeddings)
        return self.output_projection(self.normalization_layer(token_embeddings))

def create_model(config: ModelConfig, vocabulary_size: int) -> nn.Module:
    return Transformer(config, vocabulary_size)