import time
import torch
from torch.utils.data import DataLoader

from core.config import ModelConfig, save_model, create_model
from core.tokenizer import Tokenizer
from core.training import prepare_sequences, TextDataset, train_model
from core.generation import generate_text

def load_training_data(filename):
   with open(filename, 'r', encoding='utf-8') as f:
      return f.read()

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
    train_model(model, dataloader, config.epochs, config.learning_rate, device)

    print(f"Model trained in {(time.time() - start_time) / 60:.2f} minutes. Saving...")
    save_model(config, model, tokenizer, "models/tiny-llm.pth")
    
    model.eval()
    prompt = "To be or not to be"
    generated_text = generate_text(model, tokenizer, prompt, config.max_generated_length, config.temperature, device)
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    train()
