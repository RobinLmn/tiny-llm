import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.optim import AdamW
from typing import Optional, Callable, List, Tuple

def get_batch(data: List[int], batch_size: int, block_size: int, device: torch.device):
    """Generate a batch of random sequences from the data"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
    return x.to(device), y.to(device)

def get_learning_rate(iteration: int, learning_rate_warmum_iterations: int, learning_rate_decay_iterations: int, max_learning_rate: float, min_learning_rate: float) -> float:
    """Learning rate schedule with warmup and linear decay"""
    if iteration < learning_rate_warmum_iterations:
        return max_learning_rate * iteration / learning_rate_warmum_iterations
    if iteration > learning_rate_decay_iterations:
        return min_learning_rate
    
    decay_ratio = (iteration - learning_rate_warmum_iterations) / (learning_rate_decay_iterations - learning_rate_warmum_iterations)
    coeffecitient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeffecitient * (max_learning_rate - min_learning_rate)

@dataclass
class TrainingConfig:
    """Configuration for training"""
    max_iterations: int
    batch_size: int
    block_size: int
    betas: Tuple[float, float]
    max_learning_rate: float
    min_learning_rate: float
    learning_rate_warmum_iterations: int
    learning_rate_decay_iterations: int
    max_gradient_norm: float
    device: torch.device

def train_model(model: nn.Module, config: TrainingConfig, training_data: List[int], training_callback: Optional[Callable[[int, float], bool]] = None):
    """Train the model"""
    optimizer = AdamW(model.parameters(), lr=config.max_learning_rate, betas=config.betas)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    
    for iteration in range(config.max_iterations):
        lr = get_learning_rate(iteration, config.learning_rate_warmum_iterations, config.learning_rate_decay_iterations, config.max_learning_rate, config.min_learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        inputs, targets = get_batch(training_data, config.batch_size, config.block_size, config.device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        if config.max_gradient_norm < float('inf'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)

        optimizer.step()

        if training_callback is not None:
            should_stop = training_callback(iteration, loss)
            if should_stop:
                return
