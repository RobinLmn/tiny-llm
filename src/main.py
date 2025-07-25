import time
import torch
import torch.nn as nn
from typing import List

from core.generation import GenerationConfig, generate_text
from core.model import ModelConfig, create_model
from core.tokenizer import Tokenizer, TokenizerType
from core.training import TrainingConfig, train_model, get_batch
from core.utils import save_model, load_model

training_data_file = "data/shakespeare.txt"
training_split = 0.9
evaluation_iterations = 200
tokenizer_type = TokenizerType.CHARACTER
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_config = TrainingConfig(
    batch_size=64,
    block_size=256,
    max_iterations=1500,
    betas=(0.9, 0.95),
    max_learning_rate=1e-3,
    min_learning_rate=1e-4,
    learning_rate_warmum_iterations=100,
    learning_rate_decay_iterations=1500,
    max_gradient_norm=1.0,
    device=device,
)
generation_config = GenerationConfig(
    maximum_length=200,
    temperature=1.0,
    device=device,
)
model_config = ModelConfig(
    embedding_dimension=384,
    block_size=256,
    layer_number=6,
    head_number=6,
    hidden_dimension=4*384,
    dropout=0.2,
    device=device,
)

def load_training_data(filename):
   with open(filename, 'r', encoding='utf-8') as f:
      return f.read()

def create_training_callback(model_config: ModelConfig, training_config: TrainingConfig, model: nn.Module, tokenizer: Tokenizer, validation_data: List[int]):
    def callback(iteration: int, loss: float):
        if iteration % 200 != 0:
            return False

        model.eval()
        with torch.no_grad():
            for _ in range(evaluation_iterations):
                validation_inputs, validation_targets = get_batch(validation_data, training_config.batch_size, training_config.block_size, training_config.device)
                validation_logits = model(validation_inputs)
                validation_loss = nn.CrossEntropyLoss()(validation_logits.view(-1, validation_logits.size(-1)), validation_targets.view(-1))
        model.train()
        
        training_loss = loss.item() if hasattr(loss, 'item') else loss     
        print(f"Iter {iteration:5,}/{training_config.max_iterations:5,} | Training Loss: {training_loss:2.4f} | Validation Loss: {validation_loss:.4f}")
        save_model(model_config, model, tokenizer, f"models/tiny-llm-iter-{iteration}.pth")
        
        return validation_loss < 1.0

    return callback

def train():
    print("Loading data...")
    text = load_training_data(training_data_file)

    print("Building vocabulary...")
    tokenizer = Tokenizer(tokenizer_type=TokenizerType.CHARACTER)
    tokenizer.build_vocabulary(text)
    vocabulary_size = tokenizer.vocabulary_size

    tokens = tokenizer.encode(text)
    training_split_idx = int(training_split * len(tokens))
    training_data = tokens[:training_split_idx]
    validation_data = tokens[training_split_idx:]

    print(f"Training model...")
    start_time = time.time()
    model = create_model(model_config, vocabulary_size)
    callback = create_training_callback(model_config, training_config, model, tokenizer, validation_data)
    train_model(model, training_config, training_data, callback)

    print(f"Model trained in {(time.time() - start_time) / 60:.2f} minutes. Saving...")
    save_model(model_config, model, tokenizer, "models/tiny-llm.pth")
    
    model.eval()
    generated_text = generate_text(model, tokenizer, "ADAM:", generation_config)
    print(f"Generated:\n\n{generated_text}")

def test():
    model, tokenizer = load_model(model_config, "models/tiny-llm.pth")
    model.eval()
    generated_text = generate_text(model, tokenizer, "ADAM:", generation_config)
    print(f"Generated:\n\n{generated_text}")

if __name__ == "__main__":
    try:
        test()
    except KeyboardInterrupt:
        print("Shutting down...")
