import os
import sys
import json
import wandb
import random
import argparse
import tqdm as tqdm_module
from tqdm import tqdm
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from vllm import SamplingParams

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from algorithms.sft import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step
)
from utils.vllm_helper import *
from utils.rewards import r1_zero_reward_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFT on Qwen-Math model.")
    # --- Paths and Models ---
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_path", type=str, default="../data/math/sft_train.jsonl")
    parser.add_argument("--valid_path", type=str, default="../data/math/test.jsonl")
    parser.add_argument("--top_p", type=float, default=1.0,  help="Top-p sampling probability.")
    parser.add_argument("--temperature", type=float, default=1.0,  help="Sampling temperature.")
    parser.add_argument("--min_tokens", type=int, default=8,    help="Minimum number of tokens.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum number of tokens.")

    # --- Training Hyperparameters ---
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Total batch size.")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Batch size per device.")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value.")
    # --- Logistics ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--policy_device", type=str, default="cuda:0", help="Device for the training policy.")
    parser.add_argument("--eval_device",   type=str, default="cuda:1", help="Device for the vLLM evaluation instance.")
    parser.add_argument("--wandb_project", type=str, default="SFT-MATH", help="W&B project name.")

    args = parser.parse_args()

    assert args.batch_size % args.micro_batch_size == 0, "Batch size must be divisible by micro batch size."
    args.gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Enable cuDNN benchmark and TF32 for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("Loading train and valid data...")
    with open(args.train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    with open(args.valid_path, 'r') as f:
        valid_data = [json.loads(line) for line in f]
    
    # Prepare evaluation data and vLLM instance
    problems = [item["problem"]  for item in valid_data]
    solution = [item["solution"] for item in valid_data]

    # -----------------------------------------
    #  Initialize vLLM instance for evaluation
    # -----------------------------------------
    eval_vllm = init_vllm(args.model_id, args.eval_device, args.seed)
    eval_sampling_params = SamplingParams(
        temperature=args.temperature,  # 
        top_p=args.top_p,              # 
        min_tokens=args.min_tokens,    # 
        max_tokens=args.max_tokens,    # 
        stop=["<\answer>"],            # 
        include_stop_str_in_output=True
    )

    # ==========================================
    #  Run SFT Training
    # ==========================================
    print("")

    # -----------------------------------------
    #  Setup output directory and record file
    # -----------------------------------------
    output_dir = "../checkpoints/SFT4Math"
    os.makedirs(output_dir, exist_ok=True)
    record_path = os.path.join(output_dir, "record.txt")

    with open(record_path, 'w') as f:
        f.write(f"SFT Math Training Record - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

    def log_to_file(message):
        """Write message to both console and record file."""
        print(message)
        with open(record_path, 'a') as f:
            f.write(message + '\n')

    # Log training configuration
    config_info = f"""
        {'='*60}
        Training Configuration
        {'='*60}
        Model: {args.model_id}
        Epochs: {args.epochs}
        Batch Size: {args.batch_size} (micro: {args.micro_batch_size})
        Gradient Accumulation Steps: {args.gradient_accumulation_steps}
        Learning Rate: {args.lr}
        Training Samples: {len(train_data)}
        Validation Samples: {len(valid_data)}
        Output Directory: {output_dir}
        {'='*60}
    """
    log_to_file(config_info)

    run_name = f"sft_math_lr{args.lr}"
    log_to_file(f"\n> Starting SFT run: {run_name} with {len(train_data)} examples")

    wandb.init(project=args.wandb_project, name=run_name, config=args)
    wandb.define_metric("train_step")
    wandb.define_metric("valid_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("valid/*", step_metric="valid_step")

    # ---------------------------------------
    #  Initialize tokenizer and policy model
    # ---------------------------------------
    policy, tokenizer = init_policy(args.model_id, args.policy_device)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Compile model for faster training (PyTorch 2.0+)
    log_to_file("Compiling policy model with torch.compile()...")
    policy = torch.compile(policy)

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

    def collate_fn(batch):
        prompts   = [b['prompt']   for b in batch]
        responses = [b['response'] for b in batch]
        return tokenize_prompt_and_output(prompts, responses, tokenizer)
    train_loader = DataLoader(
        train_data,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True
    )

    train_step, valid_step = 0, 0
    best_acc = 0.0
    policy.train()

    # ======================= #
    # Start SFT Training Loop #
    # ======================= #
    for epoch in range(args.epochs):

        # ====== Training Logic ====== #
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", ncols=100)):
            # prepare batch data and move to device (non-blocking for async transfer)
            input_ids = batch['input_ids'].to(args.policy_device, non_blocking=True)
            labels = batch['labels'].to(args.policy_device, non_blocking=True)
            response_mask = batch['response_mask'].to(args.policy_device, non_blocking=True)

            # get log probabilities from the model and compute loss
            results = get_response_log_probs(policy, input_ids, labels, True)
            loss, metadata = sft_microbatch_train_step(
                results['log_probs'], response_mask, args.gradient_accumulation_steps,
                normalize_constant=response_mask.sum()
            )

            if (i + 1) % args.gradient_accumulation_steps == 0:
                clip_grad_norm_(policy.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                wandb.log({"train/loss": loss.item(), "train_step": train_step})
                train_step += 1

        # ====== Validation Logic ====== #
        policy.eval()
        load_policy_into_vllm_instance(policy, eval_vllm)
        output_path = f"./results/math/finetune.jsonl" if epoch == args.epochs - 1 else None
        acc = evaluate_vllm(eval_vllm, r1_zero_reward_fn, problems, solution, eval_sampling_params, output_path)

        policy.train()
        valid_step += 1
        log_to_file(f"Step {train_step}: Validation Accuracy = {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            model_save_path = os.path.join(output_dir, f"{run_name}_best.pt")
            torch.save(policy.state_dict(), model_save_path)
            log_to_file(f"New best model saved to {model_save_path} with accuracy {best_acc:.2f}%")
        wandb.log({"valid/accuracy": acc, "valid_step": valid_step})

        optimizer.zero_grad()

    del policy, optimizer, train_loader
    torch.cuda.empty_cache()
    wandb.finish()

    log_to_file("\nTraining complete.")
