import math
import torch
import torch.nn as nn

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute scaled dot-product attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V"""
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        attention_scores.masked_fill(mask == 0, -float('inf'))
    attention_weights = torch.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_weights, value)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layers"""
    def __init__(self, embedding_dimension: int, head_number: int):
        super().__init__()
        self.query_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.key_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.value_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.attention_projection = nn.Linear(embedding_dimension, embedding_dimension)
        self.head_number = head_number
        self.head_dimension = embedding_dimension // head_number

    def forward(self, token_embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply multi-head attention to token embeddings"""
        batch_size, sequence_length, embedding_dimension = token_embeddings.shape

        query = self.query_projection(token_embeddings)
        key = self.key_projection(token_embeddings)
        value = self.value_projection(token_embeddings)

        query = query.view(batch_size, sequence_length, self.head_number, self.head_dimension).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.head_number, self.head_dimension).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.head_number, self.head_dimension).transpose(1, 2)

        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dimension)

        return self.attention_projection(attention)
