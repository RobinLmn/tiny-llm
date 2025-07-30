import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Optional, Callable, Tuple

@dataclass
class TrainingConfig:
    """Configuration for training"""
    max_iterations: int
    betas: Tuple[float, float]
    max_learning_rate: float
    min_learning_rate: float
    max_gradient_norm: float
    weight_decay: float
    gradient_accumulation_steps: int
    device: torch.device

def train_model(model: nn.Module, config: TrainingConfig, dataloader: DataLoader, start_iteration: int = 0, callback: Optional[Callable[[int, float], bool]] = None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    optimizer = AdamW(model.parameters(), lr=config.max_learning_rate, betas=config.betas, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_iterations, eta_min=config.min_learning_rate)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    
    for _ in range(start_iteration):
        scheduler.step()

    for iteration in range(start_iteration, config.max_iterations):
        optimizer.zero_grad()
        
        for _ in range(config.gradient_accumulation_steps):
            inputs, targets = next(iter(dataloader))
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss = loss / config.gradient_accumulation_steps

            scaler.scale(loss).backward()

        if config.max_gradient_norm < float('inf'):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_gradient_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if callback is not None:
            should_stop = callback(iteration, loss * config.gradient_accumulation_steps)
            if should_stop:
                break
