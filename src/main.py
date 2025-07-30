import time
import torch
import torch.nn as nn
from rich.progress import Progress, TimeRemainingColumn, BarColumn, TextColumn, MofNCompleteColumn
from rich.console import Console

from core.generation import GenerationConfig, generate_text
from core.model import ModelConfig, create_model
from core.tokenizer import SubWordTokenizer
from core.training import TrainingConfig, train_model
from core.utils import save_model, load_model
from core.dataset import download_text_dataset, load_text_dataset

output_dir = "models/tiny"
dataset_name = "wikitext"
dataset_sub = "wikitext-103-v1"
processor_number = 4
evaluation_iterations = 5
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
training_config = TrainingConfig(
    max_iterations=100000,
    betas=(0.9, 0.95),
    max_learning_rate=6e-4,
    min_learning_rate=6e-5,
    max_gradient_norm=1.0,
    weight_decay=0.1,
    gradient_accumulation_steps=4,
    device=device,
)
generation_config = GenerationConfig(
    maximum_length=250,
    temperature=1.0,
    device=device,
)
model_config = ModelConfig(
    embedding_dimension=512,
    block_size=512,
    layer_number=8,
    head_number=8,
    hidden_dimension=4*512,
    dropout=0.1,
    device=device,
)

start_time = time.time()
tokenizer = SubWordTokenizer("gpt2")

def create_training_callback(model_config, training_config, model: nn.Module, validation_dataloader, progress):
    loss_fn = nn.CrossEntropyLoss()
    task_id = progress.add_task("Training", total=training_config.max_iterations)
    log_file = output_dir + "/logs.txt"

    with open(log_file, "a") as f:
        f.write(f"Model config: {model_config.embedding_dimension}d, {model_config.layer_number}L, {model_config.head_number}H\n")
        f.write("Iteration,Training_Loss,Validation_Loss\n")

    validation_loss = 0

    def callback(iteration: int, loss: float):
        nonlocal validation_loss

        if iteration % 10 == 0:
            progress.update(task_id, completed=iteration)

        if iteration % 1000 == 0:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                val_iter = iter(validation_dataloader)
                for _ in range(evaluation_iterations):
                    try:
                        inputs, targets = next(val_iter)
                    except StopIteration:
                        break
                    inputs, targets = inputs.to(training_config.device), targets.to(training_config.device)
                    logits = model(inputs)
                    total_loss += loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1)).item()
            model.train()
            validation_loss = total_loss / evaluation_iterations

        if iteration % 200 == 0:
            training_loss = loss.item() if hasattr(loss, 'item') else loss
            progress.console.print(f"Time: {(time.time() - start_time) / 60:3.1f}min | Iteration {iteration:,}/{training_config.max_iterations:,} | Training Loss: {training_loss:6.4f} | Validation Loss: {validation_loss:6.4f}")
            with open(log_file, "a") as f:
                f.write(f"{iteration},{training_loss:.6f},{validation_loss:.6f}\n")
                
        if iteration % 5000 == 0:
            torch.save(model.state_dict(), f"{output_dir}/tiny-llm-iter-{iteration}.pth")

        return False

    return callback

def train(model: nn.Module):
    print("Loading dataset...")
    training_dataloader, validation_dataloader = load_text_dataset(dataset_name, dataset_sub, batch_size, processor_number)

    print("Compiling model...")
    model = torch.compile(model)

    progress = Progress(
        TextColumn("[bold blue]Training", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        MofNCompleteColumn(),
        "•",
        TimeRemainingColumn(),
        console=Console(width=120)
    )

    callback = create_training_callback(model_config, training_config, model, validation_dataloader, progress)

    print(f"Training model...")
    start_time = time.time()

    progress.start()
    
    try:
        train_model(model, training_config, training_dataloader, callback=callback)
    except KeyboardInterrupt:
        progress.stop()
        raise KeyboardInterrupt("Training interrupted.")

    print(f"Model trained in {(time.time() - start_time) / 60:.2f} minutes. Saving...")
    save_model(model_config, model, f"{output_dir}/tiny-llm.pth")

def test():
    print("Loading model...")
    model = load_model(model_config, tokenizer, f"{output_dir}/tiny-llm.pth")
    model.eval()

    print("Generating text...")
    generated_text = generate_text(model, tokenizer, "The Titanic ", generation_config)
    print(f"Generated:\n\n{generated_text}")

if __name__ == "__main__":
    # download_text_dataset(dataset_name, dataset_sub, output_dir, processor_number)

    model = create_model(model_config, tokenizer.vocabulary_size)
    # model = load_model(model_config, tokenizer, model_filename)
    
    try:
        train(model)
        test()
    except KeyboardInterrupt:
        print("Training interrupted. Shutting down...")

