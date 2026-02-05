import os
import sys
import wandb
import random
import argparse
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from data.lm_dataset import InstructionTuningDataset
from algorithms.sft import compute_validation_loss


INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on ultrachat dataset.")
    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", type=str, default="../data/ultrachat/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="../data/ultrachat/valid.jsonl")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1,  help="Number of training epochs.")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for training.")
    parser.add_argument("--batch_size", type=int, default=64, help="Total batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=8, help="Micro batch for gradient accumulation.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    
    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training.")
    parser.add_argument("--wandb_project", type=str, default="SFT-Chat", help="W&B project name.")

    args = parser.parse_args()

    assert args.batch_size % args.micro_batch_size == 0, "Batch size must be divisible by micro batch size."
    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Enable cuDNN benchmark and TF32 for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -----------------------------------------
    #  Initialize tokenizer and model
    # -----------------------------------------
    print(f"Loading model and tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model.to(args.device)

    # Compile model for faster training (PyTorch 2.0+)
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    # ---------------------------------
    #  Create Datasets and DataLoaders
    # ---------------------------------
    print(f"\nLoading training data from: {args.train_path}")
    train_dataset = InstructionTuningDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.seq_length,
        template=INSTRUCTION_TEMPLATE,
        shuffle=True,
    )
    print(f"\nLoading validation data from: {args.valid_path}")
    val_dataset = InstructionTuningDataset(
        tokenizer=tokenizer,
        dataset_path=args.valid_path,
        seq_length=args.seq_length,
        template=INSTRUCTION_TEMPLATE,
        shuffle=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    
    run_name = f"sft_ultrachat_lr{args.lr}_seq{args.seq_length}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("valid_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("valid/*", step_metric="valid_step")


    train_step = 0
    valid_step = 0
    best_val_loss = float('inf')
    
    print(f"\n{'='*50}")
    print(f"Starting SFT Training")
    print(f"{'='*50}")
    print(f"  Model: {args.model_id}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Sequence Length: {args.seq_length}")
    print(f"  Batch Size: {args.batch_size} (micro: {args.micro_batch_size})")
    print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Training Chunks: {len(train_dataset)}")
    print(f"  Validation Chunks: {len(val_dataset)}")
    print(f"{'='*50}\n")

    
    # ----------------------------------
    #  Start Full-Parameter Fine-Tuning
    # ----------------------------------
    model.train()
    for epoch in range(args.epochs):
        print(f"\n>>> Epoch {epoch + 1}/{args.epochs}")
        
        # ====== Training ======
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=100)
        for i, batch in enumerate(progress_bar):
            # move batch to device (non-blocking for async transfer)
            input_ids = batch['input_ids'].to(args.device, non_blocking=True)
            labels = batch['labels'].to(args.device, non_blocking=True)
            # perform forward pass and compute loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

            epoch_loss += loss.item()
            num_batches += 1

            # Update weights after accumulating gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": torch.exp(loss).item(),
                    "train_step": train_step
                })
                train_step += 1

                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "ppl": f"{torch.exp(loss).item():.2f}"
                })
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"\nEpoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # ====== Validation ======
        print("\nRunning validation...")
        val_loss, val_perplexity = compute_validation_loss(model, val_loader, args.device)
        
        valid_step += 1
        print(f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        wandb.log({
            "valid/loss": val_loss,
            "valid/perplexity": val_perplexity,
            "valid_step": valid_step
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = f"../checkpoints/{run_name}.pt"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} (val_loss: {val_loss:.4f})")
        
        optimizer.zero_grad()
        model.train()


    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*50)
    
    wandb.finish()
