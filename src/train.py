import time
import torch
from torch.utils.data import DataLoader

from core.config import ModelConfig, save_model, create_model
from core.dataset import prepare_sequences, TextDataset
from core.generation import generate_text
from core.tokenizer import Tokenizer
from core.training import train_model
from core.transformer import TinyLLM

def load_training_data(filename):
   with open(filename, 'r', encoding='utf-8') as f:
      return f.read()

def create_training_callback(config: ModelConfig, model: TinyLLM, tokenizer: Tokenizer, batches: int, epochs: int):
    loss_history = []

    def callback(epoch: int, batch: int, loss: float):
        loss_val = loss.item() if hasattr(loss, 'item') else loss     
        loss_history.append(loss_val)

        if batch % 1000 == 0:
            avg_loss = sum(loss_history[-100:]) / min(100, len(loss_history))
            best_loss = min(loss_history)
            
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch:6,}/{batches:6,} | Loss: {loss_val:.4f} | Avg: {avg_loss:.4f} | Best: {best_loss:.4f}")
        
        if batch == 0 and epoch != 0:
            save_model(config, model, tokenizer, f"models/tiny-llm-epoch-{epoch}.pth")

    return callback

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = ModelConfig()

    print("Loading data...")
    training_text = load_training_data(config.training_data_file)

    print("Building vocabulary...")
    tokenizer = Tokenizer()
    tokenizer.build_vocabulary(training_text)

    print("Preparing dataset...")
    start_time = time.time()
    inputs, targets = prepare_sequences(training_text, tokenizer, config.sequence_length)
    dataset = TextDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    print(f"Dataset prepared in {(time.time() - start_time):.2f} seconds. Training model...")
    start_time = time.time()
    model = create_model(config, tokenizer.vocabulary_size, device)
    callback = create_training_callback(config, model, tokenizer, batches=len(dataloader), epochs=config.epochs)
    train_model(model, dataloader, config.epochs, config.learning_rate, device, callback)

    print(f"Model trained in {(time.time() - start_time) / 60:.2f} minutes. Saving...")
    save_model(config, model, tokenizer, "models/tiny-llm.pth")
    
    model.eval()
    prompt = "To be or not to be"
    generated_text = generate_text(model, tokenizer, prompt, config.max_generated_length, config.temperature, device)
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("Shutting down...")
