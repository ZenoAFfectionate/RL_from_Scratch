import os
import sys
import json
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

from algorithms.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    compute_validation_loss,
)
from utils.vllm_helper import *


# Template for instruction tuning (note: {response} is NOT included here)
INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on ultrachat dataset.")
    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--train_path", type=str, default="../data/ultrachat/train.jsonl")
    parser.add_argument("--valid_path", type=str, default="../data/ultrachat/valid.jsonl")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1,  help="Number of training epochs.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length (truncate longer).")
    parser.add_argument("--batch_size", type=int, default=64, help="Total batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="for gradient accumulation (reduced for memory).")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")

    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training.")
    parser.add_argument("--wandb_project", type=str, default="SFT-Chat", help="W&B project name.")
    parser.add_argument("--log_interval",  type=int, default=10,  help="Log training info.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate trained model.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")

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
    model, tokenizer = init_policy(args.model_id, args.device)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    model.gradient_checkpointing_enable()
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    # ----------------------------------
    #  Load Data and Create DataLoaders
    # ----------------------------------
    with open(args.train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(args.valid_path, 'r') as f:
        val_data = [json.loads(line) for line in f]

    # collate function that creates response_mask for pure instruction tuning
    def collate_fn(batch):
        """Format prompts and tokenize batch with response_mask"""
        prompts = [INSTRUCTION_TEMPLATE.format(instruction=item['prompt']) for item in batch]
        responses = [item['response'] for item in batch]
        result = tokenize_prompt_and_output(prompts, responses, tokenizer)
        return result

    train_loader = DataLoader(
        train_data,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Must be 0 for collate_fn with tokenizer
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Must be 0 for collate_fn with tokenizer
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    run_name = f"sft_ultrachat_lr{args.lr}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    wandb.define_metric("train_step")
    wandb.define_metric("valid_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("valid/*", step_metric="valid_step")

    # ----------------------------------------
    #  Setup output directory and record file
    # ----------------------------------------
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
        Training Configuration (Pure Instruction Tuning)
        {'='*60}
        Model: {args.model_id}
        Epochs: {args.epochs}
        Batch Size: {args.batch_size} (micro: {args.micro_batch_size})
        Gradient Accumulation Steps: {args.gradient_accumulation_steps}
        Learning Rate: {args.lr}
        Training Samples: {len(train_data)}
        Validation Samples: {len(val_data)}
        Log Interval: {args.log_interval} steps
        Output Directory: {output_dir}
        Note: Training ONLY on response tokens (response_mask)
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
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", ncols=128)
        for i, batch in enumerate(progress_bar):
            # prepare batch data and move to device (non-blocking for async transfer)
            input_ids = batch['input_ids'].to(args.device, non_blocking=True)
            labels = batch['labels'].to(args.device, non_blocking=True)
            response_mask = batch['response_mask'].to(args.device, non_blocking=True)

            # get log probabilities from the model and compute loss
            results = get_response_log_probs(model, input_ids, labels, return_token_entropy=False)
            loss, metadata = sft_microbatch_train_step(
                results['log_probs'],
                response_mask,  # Use response_mask instead of token_mask
                args.gradient_accumulation_steps,
                normalize_constant=response_mask.sum()
            )

            if (i + 1) % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # calculate actual loss for logging (undo the scaling)
                loss = loss.item() * args.gradient_accumulation_steps
                ppl = torch.exp(torch.tensor(loss)).item()
                wandb.log({"train/loss": loss, "train/perplexity": ppl, "train_step": train_step})
                progress_bar.set_postfix({"loss": f"{loss:.4f}", "ppl": f"{ppl:.2f}"})

                train_step += 1
                if train_step % args.log_interval == 0:
                    log_to_file(f"  Step {train_step}: loss={loss:.4f}, ppl={ppl:.2f}")
                # perform validation for some interval
                if train_step % args.eval_interval == 0:
                    # ======== Validation Logic ========
                    log_to_file("\nRunning Validation...")
                    model.eval()
                    val_loss, val_ppl = compute_validation_loss(model, val_loader, args.device)
                    model.train()
                    log_to_file(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

        # ======== Validation Logic ========
        log_to_file("\nRunning Validation...")
        model.eval()
        val_loss, val_ppl = compute_validation_loss(model, val_loader, args.device)
        log_to_file(f"Epoch {epoch + 1} Validation Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")

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
