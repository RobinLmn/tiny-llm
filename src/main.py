import time
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader

from core.generation import GenerationConfig, generate_text
from core.model import ModelConfig, create_model
from core.tokenizer import SubWordTokenizer
from core.training import TrainingConfig, train_model
from core.utils import save_model, load_model

dataset_name = "wikitext"
dataset_sub = "wikitext-103-v1"
processor_number = 4
evaluation_iterations = 200
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 12
training_config = TrainingConfig(
    max_iterations=100000,
    betas=(0.9, 0.95),
    max_learning_rate=3e-4,
    min_learning_rate=3e-5,
    max_gradient_norm=1.0,
    weight_decay=0.1,
    gradient_accumulation_steps=2,
    device=device,
)
generation_config = GenerationConfig(
    maximum_length=250,
    temperature=1.0,
    device=device,
)
model_config = ModelConfig(
    embedding_dimension=384,
    block_size=512,
    layer_number=6,
    head_number=6,
    hidden_dimension=4*384,
    dropout=0.2,
    device=device,
)

start_time = time.time()

tokenizer = SubWordTokenizer("gpt2")

def create_training_callback(model_config: ModelConfig, training_config: TrainingConfig, model: nn.Module, validation_dataloader: DataLoader):
    loss_fn = nn.CrossEntropyLoss()
    def callback(iteration: int, loss: float):
        training_loss = loss.item() if hasattr(loss, 'item') else loss 

        if iteration % 1000 == 0:
            model.eval()
            total_loss = 0

            with torch.no_grad():
                for batch in validation_dataloader:
                    inputs = batch["input_ids"][:, :-1].to(model_config.device)
                    targets = batch["input_ids"][:, 1:].to(model_config.device)
                    validation_logits = model(inputs)
                    total_loss += loss_fn(validation_logits.view(-1, validation_logits.size(-1)), targets.view(-1))

            model.train()
            
            validation_loss = total_loss / len(validation_dataloader)
            print(f"Time: {(time.time() - start_time) / 60:3.1f}min | Iter {iteration:5,}/{training_config.max_iterations:5,} | Training Loss: {training_loss:2.4f} | Validation Loss: {validation_loss:.4f}")
            
            if iteration % 5000 == 0:
                save_model(model_config, model, f"models/tiny-llm-iter-{iteration}.pth")
                
            return validation_loss < 1.5
        elif iteration % 200 == 0:
            print(f"Time: {(time.time() - start_time) / 60:3.1f}min | Iter {iteration:5,}/{training_config.max_iterations:5,} | Training Loss: {training_loss:2.4f}")

        return False

    return callback

def prepare_dataset():
    print("Preparing dataset...")
    dataset = load_dataset(dataset_name, dataset_sub, split="train", num_proc=processor_number)
    split_dataset = dataset.train_test_split(test_size=0.01, shuffle=True)
    training_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    def tokenize(example):
        return {"input_ids": [tokenizer.encode(text) for text in example["text"]]}
    
    training_data_tokenized = training_dataset.map(tokenize, batched=True, num_proc=processor_number, remove_columns=["text"])
    validation_data_tokenized = validation_dataset.map(tokenize, batched=True, num_proc=processor_number, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // model_config.block_size) * model_config.block_size
        result = {
            k: [t[i:i+model_config.block_size] for i in range(0, total_length, model_config.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    training_data_chunked = training_data_tokenized.map(group_texts, batched=True, remove_columns=training_data_tokenized.column_names, num_proc=processor_number)
    validation_data_chunked = validation_data_tokenized.map(group_texts, batched=True, remove_columns=training_data_tokenized.column_names, num_proc=processor_number)

    training_data_chunked.save_to_disk(f"data/{dataset_name}-{dataset_sub}-training-tokenized")
    validation_data_chunked.save_to_disk(f"data/{dataset_name}-{dataset_sub}-validation-tokenized")

def train():
    print("Loading dataset...")
    training_dataset = load_from_disk(f"data/{dataset_name}-{dataset_sub}-training-tokenized")
    validation_dataset = load_from_disk(f"data/{dataset_name}-{dataset_sub}-validation-tokenized")

    training_dataset.set_format(type="torch", columns=["input_ids"])
    validation_dataset.set_format(type="torch", columns=["input_ids"])

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    print("Compiling model...")
    model = create_model(model_config, tokenizer.vocabulary_size)
    model = torch.compile(model)
    
    print(f"Training model...")
    start_time = time.time()
    callback = create_training_callback(model_config, training_config, model, validation_dataloader)
    train_model(model, training_config, training_dataloader, callback)

    print(f"Model trained in {(time.time() - start_time) / 60:.2f} minutes. Saving...")
    save_model(model_config, model, "models/tiny-llm.pth")

def test():
    model = load_model(model_config, tokenizer, "models/tiny-llm.pth")
    model.eval()

    generated_text = generate_text(model, tokenizer, "The Titanic ", generation_config)
    print(f"Generated:\n\n{generated_text}")

if __name__ == "__main__":
    try:
        # prepare_dataset()
        train()
        test()
    except KeyboardInterrupt:
        print("Shutting down...")
