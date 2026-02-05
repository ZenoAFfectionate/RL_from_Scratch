import os
import sys
import wandb
import random
import argparse
from tqdm import tqdm
from datetime import datetime

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
from algorithms.sft import (
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_validation_loss,
)


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
    parser.add_argument("--micro_batch_size", type=int, default=8, help="for gradient accumulation.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")

    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training.")
    parser.add_argument("--wandb_project", type=str, default="SFT-Chat", help="W&B project name.")
    parser.add_argument("--log_interval", type=int, default=10, help="Log training info every N steps.")

    args = parser.parse_args()

    assert args.batch_size % args.micro_batch_size == 0, "Batch size must be divisible by micro batch size."
    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Enable cuDNN benchmark and TF32 for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')  # Faster matmul with slight precision trade-off

    # --------------------------------
    #  Initialize tokenizer and model
    # --------------------------------
    print(f"Loading model and tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model.to(args.device)
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

    # -----------------------------------------
    #  Setup output directory and record file
    # -----------------------------------------
    output_dir = "../checkpoints/SFT4Chat"
    os.makedirs(output_dir, exist_ok=True)
    record_path = os.path.join(output_dir, "record.txt")
    # 
    with open(record_path, 'w') as f:
        f.write(f"SFT Training Record - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
    
    def log_to_file(message):
        """Write message to both console and record file."""
        print(message)
        with open(record_path, 'a') as f:
            f.write(message + '\n')

    train_step = 0
    valid_step = 0
    best_val_loss = float('inf')

    config_info = f"""
        {'='*60}
        Training Configuration
        {'='*60}
        Model: {args.model_id}
        Epochs: {args.epochs}
        Sequence Length: {args.seq_length}
        Batch Size: {args.batch_size} (micro: {args.micro_batch_size})
        Gradient Accumulation Steps: {args.gradient_accumulation_steps}
        Learning Rate: {args.lr}
        Training Chunks: {len(train_dataset)}
        Validation Chunks: {len(val_dataset)}
        Log Interval: {args.log_interval} steps
        Output Directory: {output_dir}
        {'='*60}
    """
    log_to_file(config_info)


    # ----------------------------------
    #  Start Full-Parameter Fine-Tuning
    # ----------------------------------
    log_to_file("\nStarting training...")
    model.train()
    for epoch in range(args.epochs):
        log_to_file(f"\n>>> Epoch {epoch + 1}/{args.epochs}")

        # ======== Training Logic ========
        epoch_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=128)
        for i, batch in enumerate(progress_bar):
            # move batch to device (non-blocking for async transfer)
            input_ids = batch['input_ids'].to(args.device, non_blocking=True)
            labels = batch['labels'].to(args.device, non_blocking=True)

            # create mask for non-padding tokens
            token_mask = (labels != tokenizer.pad_token_id)

            # get log probabilities from the model and compute loss
            results = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            loss, metadata = sft_microbatch_train_step(
                results['log_probs'],
                token_mask,
                args.gradient_accumulation_steps,
                normalize_constant=token_mask.sum()
            )

            epoch_loss += loss.item() * args.gradient_accumulation_steps

            # update weights after accumulating gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # calculate actual loss for logging (undo the scaling)
                actual_loss = loss.item() * args.gradient_accumulation_steps
                actual_ppl = torch.exp(torch.tensor(actual_loss)).item()
                # log to wandb every 10 steps to reduce overhead
                if train_step % 10 == 0:
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/perplexity": actual_ppl,
                        "train_step": train_step
                    })
                train_step += 1

                progress_bar.set_postfix({
                    "loss": f"{actual_loss:.4f}",
                    "ppl": f"{actual_ppl:.2f}"
                })

                if train_step % args.log_interval == 0:
                    log_to_file(f"  Step {train_step}: loss={actual_loss:.4f}, ppl={actual_ppl:.2f}")

        # ======== Validation Logic ========
        log_to_file("\nRunning Validation...")
        model.eval()
        val_loss, val_perplexity = compute_validation_loss(model, val_loader, args.device)
        model.train()

        valid_step += 1
        log_to_file(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        wandb.log({
            "valid/loss": val_loss,
            "valid/perplexity": val_perplexity,
            "valid_step": valid_step
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            log_to_file(f"New best model saved to {best_model_path} (val_loss: {val_loss:.4f})")

        optimizer.zero_grad(set_to_none=True)

    final_model_path = os.path.join(output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    log_to_file(f"\nFinal model saved to {final_model_path}")

    summary = f"""
        {'='*60}
        Training Complete!
        {'='*60}
        Total Training Steps: {train_step}
        Best Validation Loss: {best_val_loss:.4f}
        Best Model: {os.path.join(output_dir, 'best_model.pt')}
        Final Model: {final_model_path}
        Record File: {record_path}
        {'='*60}
    """
    log_to_file(summary)

    wandb.finish()
