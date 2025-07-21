# Tiny LLM

A lightweight transformer-based language model implementation in PyTorch.

## Overview

This project implements a minimal Large Language Model (LLM) using PyTorch. It includes a complete training pipeline with custom tokenization, transformer architecture, and text generation capabilities.

## Features

- Custom transformer architecture with multi-head attention
- Built-in tokenizer for text processing
- Configurable model parameters (embedding dimensions, attention heads, layers)
- Training pipeline with data loading and model checkpointing
- Text generation with configurable length and sampling
- Model evaluation utilities

## Project Structure

```
tiny-llm/
├── src/
│   ├── core/
│   │   ├── attention.py      # Multi-head attention implementation
│   │   ├── config.py         # Model configuration and utilities
│   │   ├── embeddings.py     # Token and positional embeddings
│   │   ├── evaluate.py       # Model evaluation utilities
│   │   ├── generation.py     # Text generation functions
│   │   ├── tokenizer.py      # Custom tokenizer implementation
│   │   ├── training.py       # Training loop and data handling
│   │   └── transformer.py    # Main transformer model
│   └── train.py             # Training script
├── data/                    # Training data directory
├── models/                  # Saved model checkpoints
└── requirements.txt         # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RobinLmn/tiny-llm
cd tiny-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.7+

## License

Read the License file for more information.
