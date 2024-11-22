import argparse
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    AdamW,
)
from accelerate import Accelerator
from tqdm.auto import tqdm


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token if not already set

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(device)

    # Load and preprocess the dataset
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config, split="train")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Create DataLoader
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=args.per_device_train_batch_size)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=max_train_steps
    )

    # Prepare everything with Accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    model.train()
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if step % 100 == 0 and accelerator.is_local_main_process:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.detach().item()}")

    # Save the model (only on the main process)
    if accelerator.is_main_process:
        model.save_pretrained("trained_model")
        tokenizer.save_pretrained("trained_model")


if __name__ == "__main__":
    main()